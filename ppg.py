import os
import time
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from collections import deque
from sklearn.model_selection import train_test_split

# Forcer la croissance de la mémoire GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Graine pour la reproductibilité
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Politique de précision mixte
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# =========================================================
# 1. Fonctions utilitaires générales
# =========================================================

def load_dataset(dataset_name, val_split=0.1):
    """Charge un dataset et le prépare pour l'entraînement."""
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        input_shape = (28, 28, 1)
        num_classes = 10
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        input_shape = (32, 32, 3)
        num_classes = 10
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:
        raise ValueError(f"Dataset {dataset_name} non supporté.")

    y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
    x_trn, x_val, y_trn, y_val = train_test_split(
        x_train, y_train, test_size=val_split, random_state=SEED, stratify=y_train
    )
    return x_trn, y_trn, x_val, y_val, x_test, y_test, input_shape, num_classes

def apply_label_smoothing(labels, num_classes, smoothing=0.1):
    """Applique le lissage des étiquettes."""
    sm = tf.keras.utils.to_categorical(labels, num_classes)
    return sm * (1 - smoothing) + smoothing / num_classes

def create_data_generator(dataset_name):
    """Crée un générateur d'augmentation de données adapté au dataset."""
    if dataset_name == 'mnist':
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False
        )
    elif dataset_name == 'cifar10':
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True
        )
    else:
        raise ValueError(f"Dataset {dataset_name} non supporté.")

# =========================================================
# 2. Entraînement proxy rapide
# =========================================================

EARLY = callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
LR_SCHED = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)

def build_and_train_fast(cfg, x_sub, y_sub_smoothed, x_val, y_val_smoothed, input_shape, num_classes, datagen):
    """Construit et entraîne un modèle rapidement pour l'évaluation proxy."""
    inputs = layers.Input(input_shape)
    x = inputs
    for i in range(cfg['num_conv']):
        f = cfg['base_filters'] * (2 ** i)
        shortcut = x
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != f:
            shortcut = layers.Conv2D(f, 1, padding='same')(shortcut)
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        if x.shape[1] > 4:
            x = layers.MaxPooling2D()(x)
        else:
            x = layers.GlobalAveragePooling2D()(x)
            break
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(cfg['dropout_rate'])(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(cfg['dropout_rate'])(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = models.Model(inputs, out)

    opt = tf.keras.optimizers.AdamW(learning_rate=cfg['learning_rate'], weight_decay=1e-4)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    batch_size = cfg.get('batch_size', 128)
    model.fit(
        datagen.flow(x_sub, y_sub_smoothed, batch_size=batch_size),
        epochs=30,
        validation_data=(x_val, y_val_smoothed),
        callbacks=[EARLY, LR_SCHED],
        verbose=0
    )

    acc = model.evaluate(x_val, y_val_smoothed, verbose=0)[1]
    params = model.count_params()
    # Pénalité de complexité réduite et normalisée
    complexity_penalty = 1e-8 * params
    score = acc - complexity_penalty
    # Mesure de la latence (optionnelle)
    dummy = tf.constant(np.random.rand(1, *input_shape), dtype='float32')
    @tf.function
    def infer(x): return model(x, training=False)
    for _ in range(10): infer(dummy)
    t0 = time.time()
    for _ in range(100): infer(dummy)
    latency = (time.time() - t0) / 100 * 1000
    tf.keras.backend.clear_session()
    return score, acc, params, latency

# =========================================================
# 3. Fonctions de recherche locale améliorée
# =========================================================

def define_bounds(dataset_name):
    """Définit les bornes des hyperparamètres en fonction du dataset."""
    if dataset_name == 'mnist':
        return {
            'num_conv': (2, 4),
            'base_filters': (16, 64),
            'dropout_rate': (0.2, 0.5),
            'learning_rate': (1e-4, 1e-3),
            'batch_size': (64, 256)
        }
    elif dataset_name == 'cifar10':
        return {
            'num_conv': (3, 6),
            'base_filters': (32, 128),
            'dropout_rate': (0.3, 0.6),
            'learning_rate': (1e-4, 5e-4),
            'batch_size': (64, 256)
        }
    else:
        raise ValueError(f"Dataset {dataset_name} non supporté.")

def random_cfg(bounds):
    """Génère une configuration aléatoire."""
    return {
        'num_conv': random.randint(bounds['num_conv'][0], bounds['num_conv'][1]),
        'base_filters': random.choice(range(bounds['base_filters'][0], bounds['base_filters'][1] + 1, 8)),
        'dropout_rate': round(random.uniform(bounds['dropout_rate'][0], bounds['dropout_rate'][1]) / 0.05) * 0.05,
        'learning_rate': random.uniform(bounds['learning_rate'][0], bounds['learning_rate'][1]),
        'batch_size': random.choice([64, 128, 256])
    }

def adaptive_step(bounds, iteration, max_iterations):
    """Calcule des pas adaptatifs pour la recherche."""
    progress = iteration / max_iterations
    factor = 1 - 0.5 * progress  # Réduit les pas au fil du temps
    return {k: (bounds[k][1] - bounds[k][0]) * 0.1 * factor for k in bounds}

history = []
cache = {}

def record_history(cfg, score, acc, params, latency, max_len=100):
    """Enregistre l'historique des configurations."""
    history.append((cfg.copy(), score, acc, params, latency))
    if len(history) > max_len:
        history.pop(0)

def compute_correlations(bounds):
    """Calcule les corrélations entre hyperparamètres et score."""
    if len(history) < 10:
        return None
    arr = np.array([[c[k] for k in bounds] + [s] for c, s, _, _, _ in history], dtype=float)
    std = np.std(arr, axis=0)
    std[std == 0] = 1  # Éviter les divisions par zéro
    corr = np.corrcoef(arr, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)

def neighbors(cfg, steps, bounds):
    """Génère des voisins uniques pour la recherche locale."""
    out = set()
    for p in cfg:
        for d in (+steps[p], -steps[p]):
            c = cfg.copy()
            c[p] = np.clip(c[p] + d, bounds[p][0], bounds[p][1])
            if p in ['num_conv', 'base_filters', 'batch_size']:
                c[p] = int(round(c[p]))
            out.add(tuple(sorted(c.items())))
        if p in ['num_conv', 'base_filters', 'batch_size']:
            for _ in range(2):  # Générer deux candidats aléatoires par paramètre
                c = cfg.copy()
                if p == 'batch_size':
                    c[p] = random.choice([64, 128, 256])
                else:
                    c[p] = random.randint(bounds[p][0], bounds[p][1])
                out.add(tuple(sorted(c.items())))
    # Convertir les tuples en dictionnaires
    unique_neighbors = [dict(t) for t in out]
    return unique_neighbors[:10]  # Limiter à 10 candidats

def neighbors_with_blocks(cfg, corr, steps, bounds):
    """Génère des voisins en tenant compte des corrélations."""
    out = neighbors(cfg, steps, bounds)
    params = list(bounds)
    for i, p1 in enumerate(params):
        if abs(corr[-1, i]) >= 0.5:  # Seuil réduit
            c = cfg.copy()
            c[p1] = np.clip(c[p1] + steps[p1] * np.sign(corr[-1, i]), bounds[p1][0], bounds[p1][1])
            if p1 in ['num_conv', 'base_filters', 'batch_size']:
                c[p1] = int(round(c[p1]))
            out.append(c)
    # Éliminer les doublons
    unique_out = []
    seen = set()
    for c in out:
        c_tuple = tuple(sorted(c.items()))
        if c_tuple not in seen:
            seen.add(c_tuple)
            unique_out.append(c)
    return unique_out[:10]

def hill_climb(cfg, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen, bounds, max_iters=3):
    """Effectue une montée de colline locale."""
    current, curr_score, curr_acc, curr_params, curr_latency = cfg.copy(), *build_and_train_fast(
        cfg, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen
    )
    for i in range(max_iters):
        steps = adaptive_step(bounds, i, max_iters)
        cand = neighbors(current, steps, bounds)
        scored = [(build_and_train_fast(c, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen), c) for c in cand]
        best_result, best_cfg = max(scored, key=lambda x: x[0][0])
        if best_result[0] > curr_score:
            current, curr_score, curr_acc, curr_params, curr_latency = best_cfg.copy(), *best_result
        else:
            break
    return current, curr_score, curr_acc, curr_params, curr_latency

def resample_data(x_trn, y_trn, num_classes, sub_rate=0.6):
    """Rééchantillonne un sous-ensemble de données stratifié."""
    idx = np.random.choice(len(x_trn), int(sub_rate * len(x_trn)), replace=False)
    xs, ys = x_trn[idx], y_trn[idx]
    return xs, ys, apply_label_smoothing(ys, num_classes)

def evaluate_proxy(cfg, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen, num_runs=3):
    """Évalue une configuration avec plusieurs runs pour réduire la variabilité."""
    key = tuple(sorted(cfg.items()))
    if key in cache:
        return cache[key]
    scores = [build_and_train_fast(cfg, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen)
              for _ in range(num_runs)]
    avg_score = float(np.mean([s[0] for s in scores]))
    avg_acc = float(np.mean([s[1] for s in scores]))
    avg_params = float(np.mean([s[2] for s in scores]))
    avg_latency = float(np.mean([s[3] for s in scores]))
    cache[key] = (avg_score, avg_acc, avg_params, avg_latency)
    return cache[key]

def pretty_cfg(cfg):
    """Formate une configuration pour l'affichage."""
    return ', '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in cfg.items())

# =========================================================
# 4. Boucle Tabu Search + Hill Climbing
# =========================================================

def tabu_search(dataset_name, max_iter=12, sub_rate=0.6, tabu_size=15, stagnation_limit=4):
    """Effectue une recherche Tabu optimisée."""
    # Charger les données
    x_trn, y_trn, x_val, y_val, x_test, y_test, input_shape, num_classes = load_dataset(dataset_name)
    datagen = create_data_generator(dataset_name)
    bounds = define_bounds(dataset_name)

    # Préparation des données proxy
    x_sub, y_sub, y_sub_sm = resample_data(x_trn, y_trn, num_classes, sub_rate)
    datagen.fit(x_sub)
    y_val_sm = apply_label_smoothing(y_val, num_classes)

    stagnation_count = 0
    t0 = random_cfg(bounds)
    current, curr_score, curr_acc, curr_params, curr_latency = hill_climb(
        t0, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen, bounds, max_iters=3
    )
    best, best_score, best_acc, best_params, best_latency = current.copy(), curr_score, curr_acc, curr_params, curr_latency
    record_history(best, best_score, best_acc, best_params, best_latency)

    tabu = deque(maxlen=tabu_size)
    tabu.append(current)

    print("===== DÉBUT TABU SEARCH =====")
    print(f"Configuration initiale : {pretty_cfg(best)} | Score={best_score:.4f} | Acc={best_acc:.4f} | Params={best_params:.0f} | Latency={best_latency:.2f}ms\n")

    for it in range(1, max_iter + 1):
        print(f"--- Itération {it} ---")
        if it % 4 == 0:
            x_sub, y_sub, y_sub_sm = resample_data(x_trn, y_trn, num_classes, sub_rate)
            datagen.fit(x_sub)
            cache.clear()
            print("(Resampling des données proxy effectué)")

        corr = compute_correlations(bounds)
        steps = adaptive_step(bounds, it, max_iter)
        use_corr = corr is not None and np.any(np.abs(corr[-1, :-1]) >= 0.5)
        cands = neighbors_with_blocks(current, corr, steps, bounds) if use_corr else neighbors(current, steps, bounds)
        print(f"{len(cands)} candidats générés")

        if len(history) > 5 and it > 2:
            median_score = np.median([s for _, s, _, _, _ in history])
            cands = [c for c in cands if evaluate_proxy(c, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen)[0] > median_score * 0.8]

        scored = []
        for cfg in cands:
            try:
                score, acc, params, latency = evaluate_proxy(cfg, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen)
                record_history(cfg, score, acc, params, latency)
                print(f"  Voisin : {pretty_cfg(cfg)} -> Score={score:.4f} | Acc={acc:.4f} | Params={params:.0f} | Latency={latency:.2f}ms")
                scored.append((score, acc, params, latency, cfg))
            except Exception as e:
                print(f"Erreur dans l'évaluation d'un candidat : {e}")

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, acc, params, latency, cfg in scored:
            if cfg not in tabu or score > best_score:
                current, curr_score, curr_acc, curr_params, curr_latency = cfg.copy(), score, acc, params, latency
                break
        else:
            print("Aucun candidat valide trouvé, restant sur la configuration actuelle.")
            continue

        print(f"Meilleur local : {pretty_cfg(current)} | Score={curr_score:.4f} | Acc={curr_acc:.4f} | Latency={curr_latency:.2f}ms")
        if curr_score > best_score:
            print("✅ Amélioration globale !")
            best, best_score, best_acc, best_params, best_latency = current.copy(), curr_score, curr_acc, curr_params, curr_latency
            stagnation_count = 0
        else:
            print("❌ Pas d'amélioration globale.")
            stagnation_count += 1

        if stagnation_count >= stagnation_limit:
            print("⚠️ Stagnation détectée. Redémarrage aléatoire.")
            current = random_cfg(bounds)
            current, curr_score, curr_acc, curr_params, curr_latency = hill_climb(
                current, x_sub, y_sub_sm, x_val, y_val_sm, input_shape, num_classes, datagen, bounds, max_iters=3
            )
            stagnation_count = 0
            print(f"Nouvelle configuration : {pretty_cfg(current)} | Score={curr_score:.4f} | Acc={curr_acc:.4f}")

        if random.random() < 0.3:  # Augmenter l'exploration aléatoire
            score, acc, params, latency, cfg = random.choice(scored)
            current = cfg.copy()
            print(f"🔄 Exploration aléatoire : {pretty_cfg(current)} | Score={score:.4f} | Acc={acc:.4f}")

        tabu.append(current)
        print(f"Meilleur global : {pretty_cfg(best)} | Score={best_score:.4f} | Acc={best_acc:.4f} | Latency={best_latency:.2f}ms\n")

    return best, x_trn, y_trn, x_val, y_val, x_test, y_test, input_shape, num_classes, datagen

# =========================================================
# 5. Entraînement complet et sauvegarde
# =========================================================

def full_train(cfg, x_trn, y_trn, x_val, y_val, x_test, y_test, input_shape, num_classes, datagen, epochs=100):
    """Entraîne le modèle final avec la meilleure configuration."""
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    y_trn_sm = apply_label_smoothing(y_trn, num_classes)
    y_val_sm = apply_label_smoothing(y_val, num_classes)
    y_test_sm = apply_label_smoothing(y_test, num_classes)

    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    batch_size = cfg.get('batch_size', 128)
    if batch_size % num_gpus != 0:
        batch_size = (batch_size // num_gpus) * num_gpus
        print(f"Ajustement de la taille du lot à {batch_size} pour {num_gpus} GPUs")

    with strategy.scope():
        inputs = layers.Input(input_shape)
        x = inputs
        for i in range(cfg['num_conv']):
            f = cfg['base_filters'] * (2 ** i)
            shortcut = x
            x = layers.Conv2D(f, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(f, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            if shortcut.shape[-1] != f:
                shortcut = layers.Conv2D(f, 1, padding='same')(shortcut)
            x = layers.Add()([shortcut, x])
            x = layers.Activation('relu')(x)
            if x.shape[1] > 4:
                x = layers.MaxPooling2D()(x)
            else:
                x = layers.GlobalAveragePooling2D()(x)
                break
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(cfg['dropout_rate'])(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(cfg['dropout_rate'])(x)
        out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
        model = models.Model(inputs, out)

        opt = tf.keras.optimizers.AdamW(learning_rate=cfg['learning_rate'], weight_decay=1e-4)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        model.fit(
            datagen.flow(x_trn, y_trn_sm, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_val, y_val_sm),
            callbacks=[
                EARLY,
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ],
            verbose=0
        )
        acc = model.evaluate(x_test, y_test_sm, verbose=0)[1]
        return model, acc

if __name__ == '__main__':
    # Exemple d'utilisation avec MNIST ou CIFAR-10
    dataset_name = 'mnist'  # Peut être changé en 'cifar10'
    best, x_trn, y_trn, x_val, y_val, x_test, y_test, input_shape, num_classes, datagen = tabu_search(dataset_name)

    # Entraînement final
    best_model, final_acc = full_train(
        best, x_trn, y_trn, x_val, y_val, x_test, y_test, input_shape, num_classes, datagen
    )
    print(f"Meilleur modèle final -> Acc test : {final_acc:.4f} | Config : {pretty_cfg(best)} | Params : {best_model.count_params()}")

    # Latence d’inférence
    dummy = tf.constant(np.random.rand(1, *input_shape), dtype='float32')
    @tf.function
    def infer(x): return best_model(x, training=False)
    for _ in range(10):
        infer(dummy)  # Warm-up
    t0 = time.time()
    for _ in range(100):
        infer(dummy)  # Measure latency
    lat = (time.time() - t0) / 100 * 1000
    print(f"Latence inférence : {lat:.2f} ms")

    # Sauvegarde
    best_model.save(f'best_{dataset_name}_model.h5')
    with open(f'best_{dataset_name}_config.json', 'w') as f:
        json.dump({
            'config': best,
            'test_accuracy': float(final_acc),
            'parameters': int(best_model.count_params()),
            'latency_ms': float(lat)
        }, f, indent=2)
    print("Modèle et configuration sauvegardés.")