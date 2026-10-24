# Retail Customer Intelligence

Seis notebooks para un caso guiado de la materia **Explotación de Datos**. El objetivo es mostrar cómo un archivo de transacciones aparentemente utilizable se convierte, mediante decisiones de calidad, estructura y tiempo, en un resultado predictivo que puede apoyar una decisión.

## Pregunta de negocio

En cada corte mensual, ordenar clientes por la probabilidad de que realicen una compra durante los **30 días siguientes**. Los cortes de junio y julio entrenan, agosto calibra y septiembre queda como prueba temporal final.

El producto final es una lista priorizada para una campaña de retención. Las predicciones describen propensión; no prueban que una promoción cause una compra. Medir ese efecto requiere una prueba A/B posterior.

## Recorrido del caso

| Notebook | Pregunta que responde | Por qué importa |
|---|---|---|
| 01 | ¿Qué es una fila y qué problemas tiene el archivo? | Evita modelar líneas de factura como si fueran clientes. |
| 02 | ¿Qué cuenta como venta válida? | Convierte decisiones de negocio en reglas auditables. |
| 03 | ¿Qué información existía antes de decidir? | Construye perfiles y evita fuga de información. |
| 04 | ¿Qué logra el baseline crudo? | Expone la accuracy engañosa y una fuga didáctica inválida. |
| 05 | ¿Cómo se comporta el mismo modelo con perfiles procesados? | Evalúa ranking y revisa señales útiles. |
| 06 | ¿Qué se puede entregar? | Compara sólo resultados válidos y exporta una lista revisable. |

El material incluye una variable de fuga creada expresamente. Debe producir una métrica casi perfecta y **debe rechazarse**: utiliza actividad posterior al corte. Es parte de la lección, no un modelo desplegable.

## Fuente pública

El proyecto utilizará el dataset **Online Retail** de UCI (id 352), que reúne transacciones de una tienda en línea entre diciembre de 2010 y diciembre de 2011. Enlaces de referencia:

- Página del dataset: <https://archive.ics.uci.edu/dataset/352/online-retail>
- Descarga oficial: <https://archive.ics.uci.edu/static/public/352/online+retail.zip>

La descarga debe quedar cacheada como `data/raw/online_retail.zip`; los notebooks deben reutilizarla si ya existe.

## Estructura

```text
data/raw/          # Descarga original; nunca se sobrescribe ni se altera.
data/interim/      # Tablas temporales reproducibles.
data/processed/    # Datos limpios y tablas de modelado.
artifacts/         # Modelos serializados y predicciones.
reports/           # Métricas, figuras y CSV final.
notebooks/         # Los seis .ipynb finales.
src/               # Lógica compartida y reproducible.
```

## Ejecución local

Desde este directorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Ejecuta los notebooks en orden numérico. El primero descarga la fuente pública y los siguientes reutilizan los archivos producidos por las etapas anteriores.

La primera ejecución necesita internet únicamente para obtener la fuente pública; las posteriores reutilizan la caché local.
