#importar bibliotecas
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import time


#importar arquivos
df = pd.read_csv('./dataset_tratado.csv', sep=',')
df_class2 = df[df['Class'] == 2]
df_class4 = df[df['Class'] == 4]

modelo = joblib.load('../modelo.joblib')


#configuração padrão da página
st.set_page_config(page_title='Predict Cancer', layout='centered')


#descrição página principal
st.title('MODEL PREDIKSI UNTUK DETEKSI KANKER PAYUDARA')

with st.expander("Keterangan"):
    st.write("""
     
Di Brazil, kanker payudara adalah tumor ketiga dengan kejadian tertinggi, dan mungkin atau mungkin tidak berkembang lebih cepat. Karena itu, pengobatan terbaik selalu dimulai dengan diagnosis dini. Berdasarkan pernyataan ini, kami sekarang dapat membuat kecerdasan buatan yang mampu menghasilkan diagnosis yang semakin akurat untuk membantu profesional kesehatan membuat keputusan.

Dengan banyaknya data yang tersedia saat ini, kami memiliki potensi untuk melatih model pembelajaran mesin untuk menganalisis pasien berdasarkan serangkaian karakteristik dan dengan demikian menunjukkan kemungkinan adanya anomali.

     """)
    st.image("https://static.wixstatic.com/media/nsplsh_49436d2d634549466b3449~mv2.jpg/v1/fill/w_1175,h_550,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Image%20by%20National%20Cancer%20Institute.jpg")
    

#descrição barra lateral
st.sidebar.caption('# Masukkan di bawah diagnosis bak mandi karakteristik terisolasi dan kemudian membuat prediksi:')


#parâmetros do modelo
x = {
    'Ketebalan rumpun': 4.0,
    'Keseragaman Ukuran Sel': 3.0,
    'Keseragaman Bentuk Sel': 3.0,
    'Pertumbuhan Sel': 2.0,
    'Ukuran Sel Epitel Tunggal': 3.0,
    'Bare Nuclei': 3.0,
    'Bland Chromatin': 3.0,
    'Normal Nucleoli': 2.0,
    'Mitoses': 1.0
    }

for item in x:
    valor = st.sidebar.slider(label=f'{item}' ,min_value=1.0, value=x[item], max_value=10.0, step=0.01)
    #receber valores do input    
    x[item] = valor


valores_x = pd.DataFrame(x, index=[0])


#criar botao para rodar modelo
botao = st.sidebar.button('membuat prediksi')
if botao:
    #fazer previsão
    classificar = modelo.predict(valores_x)
    
    prob_predict = modelo.predict_proba(valores_x)
    class_2 = prob_predict[0][0]
    class_4 = prob_predict[0][1]
    
    #aguardar conclusão da previsão
    with st.sidebar:
        with st.spinner('Aguarde um momento...'):
            time.sleep(1)
    
    #exibir resultado
    if classificar[0] == 2:
        st.sidebar.success(
        '### Tumor classificado como Benigno!\n###### *{:.2%} de chance de não ser maligno.'.format(class_2)
                  )
    else:
        st.sidebar.error(
        '### Tumor classificado como Maligno!\n###### *{:.2%} de chance de ser maligno.'.format(class_4)
                )
        
st.sidebar.markdown("<div> <br> </div>", unsafe_allow_html=True)

#interface gráfica
st.markdown('#### Bagan Distribusi Frekuensi:')
categoria_grafico = st.selectbox('Pilih fitur yang akan dilihat:',
                                 options = ['Ketebalan rumpun', 'Keseragaman Ukuran Sel', 'Keseragaman Bentuk Sel',
                                            'Pertumbuhan Sel', 'Ukuran Sel Epitel Tunggal', 'Bare Nuclei',
                                            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'])
st.markdown('*karakteristik wanita yang dipantau dalam kelompok studi')
fig = go.Figure()
fig.add_trace(go.Histogram(x=df_class2[categoria_grafico], name='Pasien dengan Tumor Jinak', marker_color='#40A8C0'))
fig.add_trace(go.Histogram(x=df_class4[categoria_grafico], name='Pasien dengan Tumor Ganas', marker_color='#B3093F'))
fig.add_vline(x=x[categoria_grafico], line_width=3, line_dash="dash", line_color="yellow")
fig.update_traces(opacity=0.8)
fig.update_layout(yaxis_title_text='Jumlah', xaxis_title_text=categoria_grafico, barmode='overlay',
                  legend_orientation='h', legend_font_size=14, legend_y=1.13, legend_x=0)

st.plotly_chart(fig, use_container_width=True)
        
    
#rodape
