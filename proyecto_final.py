import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


# Título y descripción
st.markdown("<h2 style='text-align: center; font-size: 60px;'>Encontrando Nuevo Talento</h2>", unsafe_allow_html=True)
st.write(" ")

# Mostrar canción
cancion_inicio = """
<div style='margin-top: 50px;'>
    <div style='text-align: center;'>
<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/7nzVZu86rcXfugV9loduiA?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe></div>
"""
st.markdown(cancion_inicio, unsafe_allow_html=True)

# Mostrar iconos
iconos = """
<div style='margin-top: 100px;'>
    <div style='text-align: center;'>
        <iframe src="https://lottie.host/embed/929e7e4b-2acf-4c33-a6a5-6ffd1f5c611b/1iyt1Mjh1K.json" width="300" height="250" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
    </div>
</div>
"""
st.markdown(iconos, unsafe_allow_html=True)
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")


# Gráfico 1
# Cargar el archivo CSV
artistas_espanoles = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs/artistas_espanoles.csv')

# Crear una copia de la columna 'Géneros' y separar los géneros
generos = artistas_espanoles['Géneros'].str.split(', ').explode()
# Contar la frecuencia de cada género
genre_count = generos.value_counts()
# Filtrar los 20 géneros más repetidos
top_genres = genre_count.head(10)

# Crear una lista de colores con el mismo color por defecto
colors = ['seagreen'] * len(top_genres)

# Definir los géneros a resaltar
highlight_genres = ['Latin Pop', 'Hip-Hop/Rap']

# Cambiar el color de los géneros a resaltar
for i, genre in enumerate(top_genres.index):
    if genre in highlight_genres:
        colors[i] = 'black'  # O cualquier otro color destacado

# Crear un gráfico de barras con Plotly
fig1 = px.bar(top_genres, 
              x=top_genres.index, 
              y=top_genres.values, 
              labels={'x': 'Géneros', 'y': 'Frecuencia'}, 
              title='Top 20 Géneros Musicales Más Repetidos en Artistas Españoles en el último año')

# Aplicar los colores personalizados
fig1.update_traces(marker=dict(color=colors))

# Ajustar la apariencia del gráfico
fig1.update_layout(xaxis_tickangle=-45)  # Inclinar las etiquetas del eje x 45 grados




# Gráfico 2
# Cargar el archivo CSV para el segundo gráfico
urbano_espanol_df = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/urbano_espanol.csv')
# Seleccionar columnas de características musicales
caracteristicas_musicales = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
# Extraer los valores de las características musicales
valores_caracteristicas = urbano_espanol_df[caracteristicas_musicales].iloc[0]
# Crear un DataFrame para Plotly
df_caracteristicas = pd.DataFrame({
    'Característica': caracteristicas_musicales,
    'Valor': valores_caracteristicas
})
# Ordenar el DataFrame por el valor de las características de mayor a menor
df_caracteristicas = df_caracteristicas.sort_values(by='Valor', ascending=False)
# Crear el gráfico de barras con Plotly
fig2 = px.bar(df_caracteristicas, x='Característica', y='Valor', title='Características Musicales del Género Urbano Español', color_discrete_sequence=['seagreen'])

# Mostrar los gráficos en dos columnas
col1, col2 = st.columns([2.5, 2.25])

with col1:
    st.plotly_chart(fig1)

with col2:
    st.plotly_chart(fig2)





# Cargar los datos
data = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/top_artistas_urbano.csv')

# Gráfico 3
# Definir una función para crear y mostrar el gráfico
def mostrar_graficos():
    # Gráfico de pastel con Plotly
    pronouns_counts = data['Pronombres'].value_counts()
    color_discrete_map = {'He/Him': 'seagreen', 'She/Her': 'darksalmon'}
    
    fig_pie = px.pie(names=pronouns_counts.index, values=pronouns_counts.values, 
                     title='Distribución de Género',
                     color=pronouns_counts.index, color_discrete_map=color_discrete_map)

    # Mostrar los gráficos en Streamlit
    st.plotly_chart(fig_pie, use_container_width=True)

# Añadir un botón para mostrar ambos gráficos
if st.button('Mostrar gráficos'):
    mostrar_graficos()




# Gráficos 4 y 5
# Reemplazar los valores -1 por NaN para excluirlos del cálculo de promedios
data.replace(-1, pd.NA, inplace=True)

# Función para mostrar comparativa de seguidores entre las plataformas
def comparativa_seguidores():
    metrics = data[['Seguidores en TikTok', 'Seguidores en Spotify', 'Seguidores en Instagram']]
    st.write("### Comparativa de Seguidores entre plataformas")
    # Derretir el DataFrame para que sea adecuado para Plotly
    metrics_melted = metrics.mean().reset_index()
    metrics_melted.columns = ['Plataforma', 'Seguidores Promedio']
    
    # Definir los colores deseados para cada plataforma
    color_discrete_map = {'Seguidores en TikTok': 'khaki', 
                          'Seguidores en Spotify': 'seagreen', 
                          'Seguidores en Instagram': 'darksalmon'}
    
    fig = px.bar(metrics_melted, x='Plataforma', y='Seguidores Promedio', color='Plataforma', 
                 labels={'Seguidores Promedio': 'Seguidores Promedio', 'Plataforma': 'Plataforma'},
                 color_discrete_map=color_discrete_map)
    # Ajustar el tamaño del gráfic
    fig.update_layout(width=900, height=550)  # Ancho y alto deseado del gráfico
    st.plotly_chart(fig, use_container_width=True)

# Menú desplegable para seleccionar el tipo de métrica
tipo_metrica = st.selectbox("Selecciona el tipo de métrica:", ("Engagement por género", "Seguidores"))

# Mostrar la comparativa correspondiente según el tipo de métrica seleccionada
if tipo_metrica == "Engagement por género":
    # Gráfico 4
    # Comparativa de Engagement entre plataformas por género
    st.write("### Comparativa de Engagement entre plataformas por género")
    platforms = ['Engagement en Instagram', 'Tasa de Participación en TikTok', 'Engagement en YouTube']
    data_melted = data.melt(id_vars='Pronombres', value_vars=platforms, var_name='Plataforma', value_name='Engagement')
    data_melted_filtered = data_melted[data_melted['Engagement'].notna()]
    data_melted_filtered['Engagement'] = pd.to_numeric(data_melted_filtered['Engagement'])
    plt.figure(figsize=(8, 4))  # Cambio de tamaño aquí
    # Definir los colores para cada género
    colors = {'He/Him': 'seagreen', 'She/Her': 'darksalmon'}
    sns.barplot(x='Plataforma', y='Engagement', hue='Pronombres', data=data_melted_filtered, palette=colors, ci=None)
    plt.xlabel('Plataforma', fontsize=8)
    plt.ylabel('Engagement', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(title='Pronombres', title_fontsize='8', fontsize='7')
    st.pyplot(plt)

elif tipo_metrica == "Seguidores":
    comparativa_seguidores()




# Gráfico 6
# Crear dos columnas en Streamlit
col1, col2 = st.columns([3, 1])

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write('##### Filtrado de Artistas:')
    pronombres = st.selectbox('Selecciona pronombres', data['Pronombres'].unique())
    etapa_carrera = st.selectbox('Selecciona etapa de la carrera', data['Etapa de la carrera'].unique())

# Filtrar el DataFrame según los filtros seleccionados
filtered_df = data[(data['Pronombres'] == pronombres) & (data['Etapa de la carrera'] == etapa_carrera)]

with col1:
    if not filtered_df.empty:
        # Reemplazar NaNs con ceros en las columnas relevantes
        filtered_df[['Oyentes mensuales en Spotify', 'Tasa de Participación en TikTok']] = filtered_df[['Oyentes mensuales en Spotify', 'Tasa de Participación en TikTok']].fillna(0)
        
        # Crear el gráfico de dispersión
        fig = px.scatter(filtered_df, x='Tasa de Participación en TikTok', y='Oyentes mensuales en Spotify', color='Artista', 
                            title='Oyentes mensuales en Spotify vs. Tasa de Participación en TikTok',
                            custom_data=['Artista'])  # Agregar custom_data para identificar "Judeline"
        
        # Establecer tamaño fijo para los puntos
        fig.update_traces(marker=dict(size=20))  # Tamaño deseado de los puntos

        # Resaltar el punto de "Judeline"
        fig.update_traces(
            marker=dict(
                size=20,  # Tamaño del punto
                line=dict(width=2, color='black')  # Borde negro para resaltar
            ),
            selector=dict(customdata=['Judeline'])  # Seleccionar "Judeline"
        )

        # Ajustar el tamaño del gráfico
        fig.update_layout(width=900, height=550)  # Ancho y alto deseado del gráfico
        
        # Mostrar el gráfico
        st.plotly_chart(fig)
    else:
        st.write('No hay datos para los filtros seleccionados.')



st.write("")
st.write("")
st.write("")


# Caso Judeline
# Define the columns with adjusted ratios
col1, col2 = st.columns([1.5, 1])  # Increase the width of the first column

with col1:
    st.image('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/fotos/judeline.png', caption=' ', use_column_width=True)

# Contenido del markdown
judeline_markdown = """
<div style='margin-top: 300px;'>
    <h2 style='text-align: center; font-size: 40px;'>Judeline</h2>
"""

# Muestra el markdown en col2
with col2:
    st.markdown(judeline_markdown, unsafe_allow_html=True)


st.write("")
st.write("")
st.write("")


# Gráfico 7
# Configuración de parámetros de fuente
plt.rcParams.update({'font.size': 14})  # Cambia este valor para ajustar el tamaño de la fuente
# Datos
categorias = ['13-17', '18-24', '25-34', '35-44', '45-64', '65+']
masculino = [2.3, 22.8, 27.3, 5.4, 1.2, 0]
femenino = [3.1, 19.3, 16.1, 2.0, 0.5, 0]
# Crear el gráfico de barras apiladas
fig = go.Figure()

fig.add_trace(go.Bar(
    x=categorias,
    y=masculino,
    name='Masculino',
    marker_color='seagreen'
))

fig.add_trace(go.Bar(
    x=categorias,
    y=femenino,
    name='Femenino',
    marker_color='darksalmon'
))

# Resaltar el grupo masculino de 25 a 34 años
highlighted_masculino = ['rgba(0, 100, 0, 1)' if cat == '25-34' else 'seagreen' for cat in categorias]

fig.update_traces(
    marker=dict(
        color=highlighted_masculino,
        line=dict(
            width=2,
            color=['black' if cat == '25-34' else 'seagreen' for cat in categorias]
        )
    ),
    selector=dict(name='Masculino')
)

# Configurar el gráfico
fig.update_layout(
    barmode='stack',
    title='Distribución de la Audiencia por Edad y Género',
    xaxis_title='Grupo de Edad',
    yaxis_title='Porcentaje',
    legend_title='Género',
    height=600
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig, use_container_width=True)

# Grupo mayoritario de audiencia
# Calcular el total de audiencia para cada categoría de edad y género
audiencia_total = np.array(masculino) + np.array(femenino)

# Encontrar la categoría con la audiencia total máxima
indice_grupo_mayoritario = np.argmax(audiencia_total)

# Obtener el género correspondiente al grupo mayoritario
genero_mayoritario = 'Masculino' if masculino[indice_grupo_mayoritario] > femenino[indice_grupo_mayoritario] else 'Femenino'

# Obtener el porcentaje de audiencia del grupo mayoritario
porcentaje_mayoritario = audiencia_total[indice_grupo_mayoritario]

# Obtener la categoría de edad correspondiente al grupo mayoritario
grupo_mayoritario = categorias[indice_grupo_mayoritario]

# Crear el texto para mostrar
texto_mayoritario = f"El grupo mayoritario de audiencia es {genero_mayoritario.lower()} de {grupo_mayoritario} años, con un porcentaje del {porcentaje_mayoritario:.1f}%."

# Mostrar el texto
st.write(texto_mayoritario)



st.write("")
st.write("")
st.write("")


# Gráfico 8
# Nivel de Público General y Afinidad por Ciudad
# Datos de las ciudades
ciudades = ['Madrid', 'Barcelona', 'Seville', 'Valencia', 'Mexico City', 'Santiago', 'Málaga', 'Bogotá', 'Cádiz', 'Buenos Aires']
publico_general = [20.609139742, 11.949567441, 5.090891746, 4.55412095, 4.448146357, 3.972478785, 3.528465413, 2.946574021, 2.813960438, 2.564897067]
afinidad = [5.3, 3.9, 1.3, 1.5, 1.1, 0.9, 1.2, 0.7, 3.3, 0.6]
latitudes = [40.416775, 41.385064, 37.389092, 39.469907, 19.432608, -33.448890, 36.721273, 4.711, 36.516379, -34.603722]
longitudes = [-3.703790, 2.173404, -5.984459, -0.376288, -99.133208, -70.648270, -4.421398, -74.072092, -6.280400, -58.381592]

# Crear DataFrame
data = pd.DataFrame({
    'Ciudad': ciudades,
    'Público General': publico_general,
    'Afinidad': afinidad,
    'Latitud': latitudes,
    'Longitud': longitudes
})

# Crear el gráfico de dispersión en el mapa
fig = px.scatter_mapbox(data, lat="Latitud", lon="Longitud", hover_name="Ciudad", hover_data=["Público General", "Afinidad"],
                        size="Público General", color="Afinidad", color_continuous_scale=px.colors.cyclical.IceFire, size_max=25, zoom=3)
# Configurar Mapbox
fig.update_layout(mapbox_style="open-street-map")
# Añadir título y ajustar el tamaño
fig.update_layout(
    title="Nivel de Público General y Afinidad por Ciudad",
    height=700,
    width=900
)
# Mostrar el gráfico en Streamlit ocupando todo el ancho
st.plotly_chart(fig, use_container_width=True)





# Radios
st.write("")
st.write("")
st.write("### Estaciones de Radio Españolas")
# Definir las estaciones de radio
estaciones = [
    "RNE Radio 3",
    "iCat FM",
    "Ibiza Global Radio",
    "RNE Radio Nacional (Radio 1)",
    "Europa FM",
    "RNE Radio 4",
    "EiTB Radio Euskadi"
]
# Crear un diccionario de frecuencia de estaciones de radio
frecuencia_estaciones = {}
for estacion in estaciones:
    if estacion in frecuencia_estaciones:
        frecuencia_estaciones[estacion] += 1
    else:
        frecuencia_estaciones[estacion] = 1
# Configurar la nube de palabras
wordcloud = WordCloud(width=900, height=300, background_color='white', prefer_horizontal=0.9,
                      colormap='viridis', max_words=100, contour_width=3, contour_color='steelblue').generate_from_frequencies(frecuencia_estaciones)
# Mostrar la nube de palabras en Streamlit
st.image(wordcloud.to_array(), caption='')




# Gráfico 9
# Marcas
st.write("")
st.write("")
st.write("### Afinidad de los seguidores con Marcas")
# Directorio donde se encuentran los logotipos
logo_dir = "/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/fotos/logos/"
# Función para obtener la imagen del logo de la marca
def get_logo(brand):
    return logo_dir + brand.lower().replace(" ", "_") + ".png"
# Crear el gráfico interactivo con Plotly
def plot_bar_chart(data, color='seagreen'):
    fig = px.bar(data, 
                 x='Percent', 
                 y='Brand', 
                 hover_data=['Followers', 'Brand Affinity'], 
                 orientation='h')
    
    # Actualizar el color de las barras
    fig.update_traces(marker_color=color)
    
    fig.update_layout(
        xaxis_title="Porcentaje",
        yaxis_title="Marca",
        hovermode="closest",
    )

    for brand in data['Brand']:
        img = get_logo(brand)
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=brand,
                sizex=0.1,
                sizey=0.1,
                sizing="contain",
                opacity=0.8,
                layer="below"
            )
        )

    fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig, use_container_width=True)  # Ajuste para ocupar todo el ancho de la pantalla

# Datos de las marcas
datos = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/top_marcas.csv')

# Mostrar el gráfico en Streamlit
plot_bar_chart(datos, color='seagreen')  # El color del gráfico será seagreen





# Gráfico 10: Canciones Judeline
st.write("")
# Establecer paleta de colores
color_palette = {
    'acousticness': 'cornflowerblue',
    'danceability': 'darksalmon',
    'energy': 'moccasin',
    'instrumentalness': 'firebrick',
    'liveness': 'indigo',
    'valence': 'seagreen',
    'speechiness': 'slategray'  
}
# Cargar el CSV en un DataFrame de Pandas
canciones_judeline = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/judeline_tracks.csv')
# Seleccionar solo las primeras 10 canciones de Judeline
judeline_df = canciones_judeline.head()
# Elegir las características a mostrar en el gráfico
caracteristicas = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']
# Calcular los valores medios de cada característica
valores_medios = judeline_df[caracteristicas].mean().sort_values(ascending=False)
# Crear una lista de barras apiladas para cada característica
barras = []
for caracteristica in caracteristicas:
    barra = go.Bar(
        x=judeline_df['name'], 
        y=judeline_df[caracteristica], 
        name=caracteristica,
        marker_color=color_palette[caracteristica]
    )
    barras.append(barra)
# Configurar el diseño del gráfico de las canciones de Judeline
layout = go.Layout(
    title="Características musicales de las canciones de Judeline",
    xaxis=dict(title='Canción'),
    yaxis=dict(title='Valor'),
    barmode='stack',  # Apilar las barras
    legend=dict(x=1, y=1, bgcolor='rgba(255, 255, 255, 0.5)'),  # Colocar la leyenda en la esquina superior derecha
    margin=dict(l=40, r=80, t=40, b=40)
)
fig_judeline = go.Figure(data=barras, layout=layout)



# Gráfico 11: Tendencias de Características de Sonido a lo largo del Tiempo
# Definir una paleta de colores para las características
st.write("")
# Características musicales a lo largo de los años
years = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/data/data_by_year.csv')
# Lista de características de sonido
sound_features = ['acousticness', 'danceability', 'energy', 'liveness', 'valence', 'speechiness']
# Crear el gráfico con Plotly para las tendencias de características de sonido a lo largo del tiempo
fig_years = go.Figure()
for feature in sound_features:
    fig_years.add_trace(go.Scatter(x=years['year'], y=years[feature], mode='lines', name=feature, line=dict(color=color_palette[feature])))
# Actualizar el diseño del gráfico
fig_years.update_layout(
    title='Tendencias de Características de Sonido a lo largo del Tiempo',
    xaxis=dict(title='Año'),
    yaxis=dict(title='Valor'),
    legend=dict(x=1, y=1),  # Ajustar la posición de la leyenda para que se mueva con el gráfico
    margin=dict(l=100, r=40, t=40, b=40)  # Incrementar el margen izquierdo para mover el gráfico a la derecha
)
# Mostrar los gráficos en Streamlit en columnas
col1, col2 = st.columns(2)

with col2:
    st.plotly_chart(fig_years)

with col1:
    st.plotly_chart(fig_judeline)






st.write("")
st.write("")
st.write("")


# Gráfico 12: Predicción de Acousticness
# Seleccionar las columnas relevantes
X = years[['year']]
y_acousticness = years['acousticness']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_ac, X_test_ac, y_train_ac, y_test_ac = train_test_split(X, y_acousticness, test_size=0.2, random_state=42)

# Crear el modelo
model_acousticness = LinearRegression()

# Entrenar el modelo
model_acousticness.fit(X_train_ac, y_train_ac)

# Predecir los valores en el conjunto de prueba
y_pred_acousticness = model_acousticness.predict(X_test_ac)

# Calcular el error medio absoluto (MAE)
mae_acousticness = np.mean(np.abs(y_test_ac - y_pred_acousticness))

# Crear un rango de años futuros para la predicción
future_years_ac = np.arange(2024, 2040).reshape(-1, 1)

# Predecir los valores futuros
future_predictions_acousticness = model_acousticness.predict(future_years_ac)

# Crear un DataFrame para las predicciones futuras
future_df_acousticness = pd.DataFrame({
    'year': future_years_ac.flatten(),
    'acousticness': future_predictions_acousticness
})
# Concatenar datos históricos y predicciones de acousticness
df_combined_acousticness = pd.concat([years[['year', 'acousticness']], future_df_acousticness], ignore_index=True)
# Graficar los datos utilizando Plotly
fig_acousticness = px.scatter(df_combined_acousticness, x='year', y='acousticness', title='Predicción de Acousticness en el Futuro')
fig_acousticness.update_traces(marker=dict(color='seagreen'), selector=dict(mode='markers'))
fig_acousticness.add_scatter(x=future_years_ac.flatten(), y=future_predictions_acousticness, mode='lines', name='Predicción', line=dict(color='red'))

# Ajustar la posición del gráfico a la izquierda
fig_acousticness.update_layout(margin=dict(l=50, r=80, t=40, b=40))




# Gráfico 13: Predicción de Danceability
# Seleccionar las columnas relevantes
y_danceability = years['danceability']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_da, X_test_da, y_train_da, y_test_da = train_test_split(X, y_danceability, test_size=0.2, random_state=42)

# Crear el modelo
model_danceability = LinearRegression()

# Entrenar el modelo
model_danceability.fit(X_train_da, y_train_da)

# Predecir los valores en el conjunto de prueba
y_pred_danceability = model_danceability.predict(X_test_da)

# Calcular el error medio absoluto (MAE)
mae_danceability = np.mean(np.abs(y_test_da - y_pred_danceability))

# Crear un rango de años futuros para la predicción
future_years_da = np.arange(2024, 2040).reshape(-1, 1)

# Predecir los valores futuros
future_predictions_danceability = model_danceability.predict(future_years_da)

# Crear un DataFrame para las predicciones futuras
future_df_danceability = pd.DataFrame({
    'year': future_years_da.flatten(),
    'danceability': future_predictions_danceability
})

# Concatenar datos históricos y predicciones de danceability
df_combined_danceability = pd.concat([years[['year', 'danceability']], future_df_danceability], ignore_index=True)

# Graficar los datos utilizando Plotly
fig_danceability = px.scatter(df_combined_danceability, x='year', y='danceability', title='Predicción de Danceability en el Futuro')
fig_danceability.update_traces(marker=dict(color='seagreen'), selector=dict(mode='markers'))
fig_danceability.add_scatter(x=future_years_da.flatten(), y=future_predictions_danceability, mode='lines', name='Predicción', line=dict(color='red'))

# Ajustar la posición del gráfico a la izquierda
fig_danceability.update_layout(margin=dict(l=90, r=90, t=40, b=40))

# Mostrar los gráficos en Streamlit en columnas
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_acousticness)

with col2:
    st.plotly_chart(fig_danceability)



# Gráfico 14: Predicción de Energy
# Seleccionar las columnas relevantes
X = years[['year']]
y = years['energy']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir los valores en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error medio absoluto (MAE)
mae = np.mean(np.abs(y_test - y_pred))

# Crear un rango de años futuros para la predicción
future_years = np.arange(2024, 2040).reshape(-1, 1)

# Predecir los valores futuros
future_predictions_energy = model.predict(future_years)

# Crear un DataFrame para las predicciones futuras
future_df = pd.DataFrame({
    'year': future_years.flatten(),
    'energy': future_predictions_energy
})

# Concatenar datos históricos y predicciones de energía
df_combined = pd.concat([years[['year', 'energy']], future_df], ignore_index=True)

# Graficar los datos utilizando Plotly
fig_energy = px.scatter(df_combined, x='year', y='energy', title='Predicción de Energy en el Futuro')
fig_energy.update_traces(marker=dict(color='seagreen'), selector=dict(mode='markers'))
fig_energy.add_scatter(x=future_years.flatten(), y=future_predictions_energy, mode='lines', name='Predicción', line=dict(color='red'))


# Ajustar la posición del gráfico a la izquierda
fig_energy.update_layout(margin=dict(l=50, r=80, t=40, b=40))



# Gráfico 15: Predicción de Valence
# Seleccionar las columnas relevantes
y_valence = years['valence']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_valence, X_test_valence, y_train_valence, y_test_valence = train_test_split(X, y_valence, test_size=0.2, random_state=42)

# Crear el modelo
model_valence = LinearRegression()

# Entrenar el modelo
model_valence.fit(X_train_valence, y_train_valence)

# Predecir los valores en el conjunto de prueba
y_pred_valence = model_valence.predict(X_test_valence)

# Calcular el error medio absoluto (MAE)
mae_valence = np.mean(np.abs(y_test_valence - y_pred_valence))

# Crear un rango de años futuros para la predicción
future_years_valence = np.arange(2024, 2040).reshape(-1, 1)

# Predecir los valores futuros
future_predictions_valence = model_valence.predict(future_years_valence)

# Crear un DataFrame para las predicciones futuras
future_df_valence = pd.DataFrame({
    'year': future_years_valence.flatten(),
    'valence': future_predictions_valence
})

# Concatenar datos históricos y predicciones de valence
df_combined_valence = pd.concat([years[['year', 'valence']], future_df_valence], ignore_index=True)

# Graficar los datos utilizando Plotly
fig_valence = px.scatter(df_combined_valence, x='year', y='valence', title='Predicción de Valence en el Futuro')
fig_valence.update_traces(marker=dict(color='seagreen'), selector=dict(mode='markers'))
fig_valence.add_scatter(x=future_years_valence.flatten(), y=future_predictions_valence, mode='lines', name='Predicción', line=dict(color='red'))

# Ajustar la posición del gráfico a la izquierda
fig_valence.update_layout(margin=dict(l=90, r=90, t=40, b=40))

# Mostrar los gráficos en Streamlit en columnas
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_energy)

with col2:
    st.plotly_chart(fig_valence)


st.write("")
st.write("")
st.write("")



# Recomendaciones de canciones de artistas emergentes
# CSVs necesarios
canciones_juicybae = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/juicy_bae_tracks.csv')
canciones_beapelea = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/bea_pelea_tracks.csv')
canciones_anier = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/anier_tracks.csv')
canciones_ani_queen = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/ani_queen_tracks.csv')
canciones_aleesha = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/aleesha_tracks.csv')
canciones_lunaki = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/lunaki_tracks.csv')
canciones_kenya_racaile = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/kenya_racaile_tracks.csv')
canciones_la_blondie = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/la_blondie_tracks.csv')
canciones_amore = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/amore_tracks.csv')
canciones_xina_mora = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/xina_mora_tracks.csv')
canciones_deva = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/deva_tracks.csv')
canciones_ly_raine = pd.read_csv('/Users/natalialabrador/Desktop/Ironhack/Proyecto Final/csvs_tableau_urbano/ly_raine_tracks.csv')

# Concatenar los DataFrames
df_combined = pd.concat([canciones_judeline, canciones_juicybae, canciones_beapelea, canciones_anier, canciones_ani_queen, canciones_aleesha, 
                         canciones_lunaki, canciones_kenya_racaile, canciones_la_blondie, canciones_amore, canciones_xina_mora, canciones_deva,
                         canciones_ly_raine
                         ], ignore_index=True)

# Preprocesamiento de datos
features_to_scale = df_combined.drop(columns=['name', 'Artista'])
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_to_scale)
scaler = StandardScaler()
df_combined_scaled = scaler.fit_transform(features_imputed)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_combined_scaled)
cluster_labels_combined = kmeans.labels_
df_combined['cluster'] = cluster_labels_combined

# Definir función de recomendación con depuración
def recomendar_canciones_otro_artista(cancion, num_recomendaciones=5):
    try:
        # Obtener el artista y el clúster de la canción seleccionada
        artista_cancion = df_combined[df_combined['name'] == cancion]['Artista'].values[0]
        cluster = df_combined[df_combined['name'] == cancion]['cluster'].values[0]

        # Filtrar las canciones en el mismo clúster y que no sean del mismo artista
        recomendaciones = df_combined[(df_combined['cluster'] == cluster) & (df_combined['Artista'] != artista_cancion)]
        
        # Verificar si hay suficientes canciones para recomendar
        if recomendaciones.empty:
            print(f"No se encontraron recomendaciones para la canción '{cancion}' en el mismo clúster.")
            return []
        
        # Tomar una muestra de las recomendaciones
        num_recomendaciones = min(num_recomendaciones, len(recomendaciones))
        recomendaciones = recomendaciones.sample(num_recomendaciones, replace=False)
        
        # Obtener el nombre de la canción y el nombre del artista para cada recomendación
        recomendaciones_info = [(row['name'], row['Artista']) for idx, row in recomendaciones.iterrows()]
        
        return recomendaciones_info
    
    except IndexError:
        print(f"La canción '{cancion}' no se encontró en el DataFrame.")
        return []


# App Streamlit con depuración
st.title('Artistas Emergentes Similares')

# Obtener la lista de canciones de Judeline
canciones_judeline = df_combined[df_combined['Artista'] == 'Judeline']['name'].unique()

# Widget de selección de canción
cancion_seleccionada = st.selectbox('Selecciona una canción:', canciones_judeline)


# Mostrar los gráficos en Streamlit en columnas
col1, col2 = st.columns(2)

with col1:
    # Obtener y mostrar las recomendaciones
    recomendaciones = recomendar_canciones_otro_artista(cancion_seleccionada)
    st.write(f"Recomendaciones para la canción '{cancion_seleccionada}':")
    if recomendaciones:
        for i, (cancion_recomendada, artista_recomendado) in enumerate(recomendaciones, start=1):
            st.write(f"{i}. '{cancion_recomendada}' por {artista_recomendado}")
    else:
        st.write(f"No se encontraron recomendaciones para la canción '{cancion_seleccionada}' en el mismo clúster.")

with col2:
    spotify_profile_markdown = """
    <div style='margin-top: 30px;'>
        <div style='text-align: center;'>
            <iframe src="https://open.spotify.com/embed/playlist/6RvV3mrph7vYsaNHhXnaNm?utm_source=generator&theme=0" width="100%" height="352" frameborder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
        </div>
    </div>
    """
    st.markdown(spotify_profile_markdown, unsafe_allow_html=True)





st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
# App Streamlit con depuración
st.title('Next Steps')
# Mostrar los logos en Streamlit en columnas
logos = """
<div style='margin-top: 150px;'>
    <div style='text-align: center;'>
        <iframe src="https://lottie.host/embed/62c7b0f8-9323-4486-8244-767551a74ccb/VfRmP91A8o.json" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
        <iframe src="https://lottie.host/embed/89dac5ea-78c1-48a0-aa60-b6212e5e7c34/ga93nnHGjl.json" width="200" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
        <iframe src="https://lottie.host/embed/54fc8dbb-8680-492e-a92f-9a9018d9408d/sABdxMpXOk.json" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
    </div>
</div>
"""
st.markdown(logos, unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

spotify_profile_markdown = """
<div style='margin-top: 150px;'>
    <div style='text-align: center;'>
        <iframe src="https://open.spotify.com/embed/artist/1H6X7yhnXZg73f9bssaj1Q" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
    </div>
</div>
"""
st.markdown(spotify_profile_markdown, unsafe_allow_html=True)












