# Mundial 2026 — Modelo de predicción e imágenes libres

Sistema de predicción del Mundial 2026 que alimenta la web
[canaldocente.es/mundial](https://canaldocente.es/mundial/).

## Qué hace

1. **Modelo de predicción** (`model/predict.py`): simulación Monte Carlo del
   torneo completo (12 grupos, mejores terceros, dieciseisavos → final).
   La fuerza de cada selección combina su **Elo histórico** (eloratings.net,
   que codifica todo el histórico de enfrentamientos internacionales), la
   **calidad del once titular**, la experiencia mundialista, la localía de los
   anfitriones, el clima/viajes y la presión sobre los favoritos. Los goles se
   generan con un proceso de Poisson. Antes de simular descarga los
   **resultados reales ya jugados** del API de canaldocente.es y condiciona
   todas las probabilidades a ellos.

   Salida `predictions.json`:
   - `champion`: probabilidad de ganar el Mundial por selección
   - `reach`: probabilidad de alcanzar cada ronda (R32 → Final)
   - `slots`: para cada cruce concreto del cuadro, probabilidad de **estar**
     en ese cruce y de **pasar** de ronda
   - probabilidad de partidos concretos a partir de `slots.advance`

2. **Imágenes libres de derechos** (`images/fetch_images.py`): para cada
   jugador del catálogo del álbum (Panini Adrenalyn XL 2026, `data/catalog.json`)
   busca su entidad en **Wikidata**, valida que es futbolista (P106=Q937857),
   recupera su fotografía (P18) alojada en **Wikimedia Commons** y la licencia
   y autor para cumplir la atribución de las licencias Creative Commons.
   Salida: `images.json` (la web la consume directamente).

3. **Automatización** (`.github/workflows/update.yml`): cada 6 horas descarga
   los resultados reales y regenera `predictions.json`; las imágenes se
   actualizan bajo demanda (workflow_dispatch).

## Fuentes y créditos

- Pipeline de referencia y datos de entrenamiento: dataset de Kaggle
  [rauffauzanrambe/fifa-world-cup-2026-prediction-system](https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-prediction-system)
  (copia en `kaggle/`).
- Ratings históricos: [eloratings.net](https://www.eloratings.net/).
- Catálogo del álbum: checklist oficial Panini Adrenalyn XL FIFA World Cup 2026.
- Fotografías: Wikimedia Commons, licencias Creative Commons con atribución
  (autor y licencia incluidos en `images.json`).

## Uso

```bash
python model/predict.py --fetch --sims 10000
python images/fetch_images.py
```
