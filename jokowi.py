import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Memuat data
data_path = "publik persepsi.csv"  # Path ke file data Anda
data = pd.read_csv(data_path)

# Menghapus kolom yang tidak diperlukan
data = data.drop(columns=['confidence', 'text_word_count', 'word_count'], errors='ignore')
st.write("### Pranala Data", data.head())
st.write("### 10 Media Teratas Berdasarkan Jumlah Publikasi")

# Menghitung jumlah media unik
jumlah_media_unik = data['Nama Media'].nunique()

# Memilih kolom yang terkait dengan 'Nama Media' dan menghitung jumlah kemunculannya
top_media = data['Nama Media'].value_counts().head(10).reset_index()
top_media.columns = ['Nama Media', 'Jumlah Publikasi']

# Membuat grafik batang menggunakan Plotly untuk Top Media
fig = px.bar(top_media, x='Nama Media', y='Jumlah Publikasi', 
             labels={'Jumlah Publikasi': 'Jumlah Publikasi', 'Nama Media': 'Media'},
             color='Jumlah Publikasi',
             color_continuous_scale='Viridis')

# Layout Streamlit dengan Kolom
col1, col2 = st.columns([2, 1])

# Kolom 1: Menampilkan Grafik Top Media
with col1:
    st.plotly_chart(fig, use_container_width=True)

# Kolom 2: Menampilkan Informasi Jumlah Media Unik
with col2:
    st.markdown(
        f"""
        <style>
            .centered-left-content {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                height: 80vh;
                padding-left: 20px;
                text-align: left;
                color: #333;  
            }}
            h3 {{
                font-size: 24px;
                font-weight: bold;
                margin: 0;
            }}
            .jumlah-media {{
                font-size: 30px;
                font-weight: bold;
                color: #4CAF50;
                margin: 5px 0;
            }}
            .deskripsi {{
                margin-top: 10px;
                font-size: 16px;
                color: #666;
            }}
        </style>
        <div class="centered-left-content">
            <h3>Jumlah Media:</h3>
            <p class="jumlah-media">{jumlah_media_unik}</p>
            <p class="deskripsi">Total jumlah media unik yang memberitakan terkait data ini.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("### Jumlah Publikasi per Bulan")
# Pastikan 'Waktu Terbit' diubah menjadi tipe datetime
data['Waktu Terbit'] = pd.to_datetime(data['Waktu Terbit'], errors='coerce')

# Menghitung jumlah publikasi per bulan
publikasi_per_bulan = data.resample('M', on='Waktu Terbit').size().reset_index(name='Jumlah Publikasi')

# Membuat grafik garis menggunakan Plotly
fig_waktu = px.line(publikasi_per_bulan, x='Waktu Terbit', y='Jumlah Publikasi',
                    labels={'Waktu Terbit': 'Waktu Terbit', 'Jumlah Publikasi': 'Jumlah Publikasi'},
                    markers=True)

# Menampilkan Grafik Jumlah Publikasi Per Bulan
st.plotly_chart(fig_waktu, use_container_width=True)

# Menghitung jumlah berita untuk setiap kategori
count_jokowi = data['Judul Berita'].str.contains('jokowi', case=False, na=False).sum()  # Mengandung 'jokowi'
count_presiden = data['Judul Berita'].str.contains('presiden', case=False, na=False).sum()  # Mengandung 'presiden'
count_presiden_jokowi = data[data['Judul Berita'].str.contains('jokowi', case=False, na=False) & 
                             data['Judul Berita'].str.contains('presiden', case=False, na=False)].shape[0]  # Mengandung keduanya
count_neither = data[~data['Judul Berita'].str.contains('jokowi', case=False, na=False) & 
                     ~data['Judul Berita'].str.contains('presiden', case=False, na=False)].shape[0]  # Tidak mengandung keduanya

st.write("### Diagram Venn")
# Membuat diagram Venn
plt.figure(figsize=(10, 6))
venn = venn2(subsets=(count_jokowi, count_presiden, count_presiden_jokowi), 
              set_labels=('"Presiden"', '"Jokowi"'))

# Mengubah warna teks untuk semua elemen
for text in venn.set_labels:
    text.set_color('white')  # Set warna label menjadi putih
for text in venn.subset_labels:
    if text is not None:  # Pastikan tidak ada label yang None
        text.set_color('white')  # Set warna subset label menjadi putih

plt.text(-0.6, 0.4, count_neither, fontsize=10, ha='center', color='white')  # Set color to white

# Simpan diagram Venn ke dalam file gambar dengan latar belakang transparan
plt.savefig('venn_diagram.png', bbox_inches='tight', transparent=True)

# Tampilkan gambar dalam Streamlit di tengah
st.image('venn_diagram.png', use_column_width='always', caption='Diagram Venn: Berita Mengandung Kata "Presiden" dan "Jokowi"')

# Tampilkan tabel untuk jumlah artikel berdasarkan Detected Language
language_counts = data['Detected Language'].value_counts().reset_index()
language_counts.columns = ['Bahasa', 'Jumlah Publikasi']

# Tampilkan tabel
st.write("### Jumlah Publikasi Berdasarkan Bahasa")
st.table(language_counts)

st.write("### Panjang Teks dalam Berbagai Bahasa Menggunakan Stopword")
# Membuat salinan data yang difilter untuk bahasa Inggris ('en') dan bahasa Indonesia ('id')
english_data = data[data['Detected Language'] == 'en'].copy()
indonesian_data = data[data['Detected Language'] == 'id'].copy()

# Menghitung panjang (jumlah kata) di setiap teks pada kolom 'Clean Text' pada salinan data
english_data['word_count'] = english_data['stopword'].apply(lambda x: len(str(x).split()))
indonesian_data['word_count'] = indonesian_data['stopword'].apply(lambda x: len(str(x).split()))

# Membuat trace untuk bahasa Inggris
trace0 = go.Box(
    y=english_data['word_count'],
    name='Teks Bahasa Inggris',
    marker=dict(
        color='blue',
    )
)

# Membuat trace untuk bahasa Indonesia
trace1 = go.Box(
    y=indonesian_data['word_count'],
    name='Teks Bahasa Indonesia',
    marker=dict(
        color='green',
    )
)

# Data untuk plot
data_plot = [trace0, trace1]

# Layout
layout = go.Layout(
    yaxis=dict(
        title="Jumlah Kata",
    ),
    xaxis=dict(
        title="Bahasa",
    )
)

# Membuat figure
fig = go.Figure(data=data_plot, layout=layout)

# Menampilkan plot di Streamlit
st.plotly_chart(fig, use_container_width=True)

# Menggabungkan semua teks menjadi satu string untuk word cloud
english_data = data[data['Detected Language'] == 'en']['stopword']
indonesian_data = data[data['Detected Language'] == 'id']['stopword']
text_english = ' '.join(english_data)
text_indonesian = ' '.join(indonesian_data)

# Fungsi untuk membuat dan menyimpan word cloud sebagai PNG
def create_wordcloud(text, lang):
    wordcloud = WordCloud(width=800, height=400, background_color=None, mode="RGBA").generate(text)
    
    # Menyimpan word cloud sebagai PNG
    filename = f'wordcloud_{lang}.png'
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    
    return filename

# Membuat dan menyimpan word cloud untuk bahasa Inggris
english_wordcloud_file = create_wordcloud(text_english, 'en')
st.write("### Word Cloud - Teks Bahasa Inggris")
st.image(english_wordcloud_file, use_column_width=True)

# Membuat dan menyimpan word cloud untuk bahasa Indonesia
indonesian_wordcloud_file = create_wordcloud(text_indonesian, 'id')
st.write("### Word Cloud - Teks Bahasa Indonesia")
st.image(indonesian_wordcloud_file, use_column_width=True)

# Menghitung jumlah data berdasarkan kelas sentimen
sentiment_counts = data['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'Count']

# Membuat visualisasi bar chart menggunakan Plotly
fig_sentiment = px.bar(sentiment_counts, x='sentiment', y='Count', color='sentiment', 

             title='Total Setiap Kelas Sentimen', 

             labels={'Count':'Total', 'Sentiment':'Sentiment Class'}, 

             height=400, width=600)

# Menampilkan grafik
st.plotly_chart(fig_sentiment, use_container_width=True)

# Filter data untuk setiap sentimen
negative_data = data[data['sentiment'] == 'negative']
positive_data = data[data['sentiment'] == 'positive']
neutral_data = data[data['sentiment'] == 'neutral']

# Menghitung jumlah berita per media untuk masing-masing sentimen
media_negative_counts = negative_data['Nama Media'].value_counts().reset_index()
media_positive_counts = positive_data['Nama Media'].value_counts().reset_index()
media_neutral_counts = neutral_data['Nama Media'].value_counts().reset_index()

# Menamai kolom hasil perhitungan
media_negative_counts.columns = ['Nama Media', 'Jumlah Berita Negatif']
media_positive_counts.columns = ['Nama Media', 'Jumlah Berita Positif']
media_neutral_counts.columns = ['Nama Media', 'Jumlah Berita Netral']

# Menampilkan hanya 10 media teratas
top_10_negative_media = media_negative_counts.head(10)
top_10_positive_media = media_positive_counts.head(10)
top_10_neutral_media = media_neutral_counts.head(10)

# Membuat visualisasi bar chart untuk Sentimen Negatif
fig_neg = px.bar(top_10_negative_media, 
                 x='Nama Media', 
                 y='Jumlah Berita Negatif', 
                 title='Top 10 Media yang Memberitakan Sentimen Negatif',
                 labels={'Nama Media':'Media', 'Jumlah Berita Negatif':'Jumlah Berita'},
                 color='Nama Media',
                 height=500)

# Membuat visualisasi bar chart untuk Sentimen Positif
fig_pos = px.bar(top_10_positive_media, 
                 x='Nama Media', 
                 y='Jumlah Berita Positif', 
                 title='Top 10 Media yang Memberitakan Sentimen Positif',
                 labels={'Nama Media':'Media', 'Jumlah Berita Positif':'Jumlah Berita'},
                 color='Nama Media',
                 height=500)

# Membuat visualisasi bar chart untuk Sentimen Netral
fig_neutral = px.bar(top_10_neutral_media, 
                     x='Nama Media', 
                     y='Jumlah Berita Netral', 
                     title='Top 10 Media yang Memberitakan Sentimen Netral',
                     labels={'Nama Media':'Media', 'Jumlah Berita Netral':'Jumlah Berita'},
                     color='Nama Media',
                     height=500)

# Menampilkan plot
st.plotly_chart(fig_neg, use_container_width=True)
st.plotly_chart(fig_pos, use_container_width=True)
st.plotly_chart(fig_neutral, use_container_width=True)

# Fungsi untuk membuat dan menyimpan word cloud
def plot_wordcloud(text, filename):
    # Menghasilkan word cloud
    wordcloud = WordCloud(width=800, height=400, background_color=None, mode="RGBA").generate(text)

    # Menampilkan word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Menyembunyikan sumbu
    plt.savefig(filename, bbox_inches='tight', transparent=True)  # Menyimpan sebagai PNG dengan latar belakang transparan
    plt.close()  # Menutup plot untuk menghindari tampilan ganda di Streamlit

# Filter data hanya untuk bahasa Inggris
english_data = data[data['Detected Language'] == 'en']

# Menggabungkan semua teks berdasarkan sentimen untuk bahasa Inggris
positive_text_en = ' '.join(english_data[english_data['sentiment'] == 'positive']['stopword'])
negative_text_en = ' '.join(english_data[english_data['sentiment'] == 'negative']['stopword'])
neutral_text_en = ' '.join(english_data[english_data['sentiment'] == 'neutral']['stopword'])

# Menampilkan dataframe untuk setiap kategori sentimen bahasa Inggris
st.write("### DataFrame untuk Sentimen Positif (Bahasa Inggris)")
st.dataframe(english_data[english_data['sentiment'] == 'positive'].drop(columns=['Clean Text'], errors='ignore'))
plot_wordcloud(positive_text_en, 'positive_sentiment_en.png')
st.image('positive_sentiment_en.png', use_column_width=True)

st.write("### DataFrame untuk Sentimen Negatif (Bahasa Inggris)")
st.dataframe(english_data[english_data['sentiment'] == 'negative'].drop(columns=['Clean Text'], errors='ignore'))
plot_wordcloud(negative_text_en, 'negative_sentiment_en.png')
st.image('negative_sentiment_en.png', use_column_width=True)

st.write("### DataFrame untuk Sentimen Netral (Bahasa Inggris)")
st.dataframe(english_data[english_data['sentiment'] == 'neutral'].drop(columns=['Clean Text'], errors='ignore'))
plot_wordcloud(neutral_text_en, 'neutral_sentiment_en.png')
st.image('neutral_sentiment_en.png', use_column_width=True)

# Filter data hanya untuk bahasa Indonesia
indonesian_data = data[data['Detected Language'] == 'id']

# Menggabungkan semua teks berdasarkan sentimen untuk bahasa Indonesia
positive_text_id = ' '.join(indonesian_data[indonesian_data['sentiment'] == 'positive']['stopword'])
negative_text_id = ' '.join(indonesian_data[indonesian_data['sentiment'] == 'negative']['stopword'])
neutral_text_id = ' '.join(indonesian_data[indonesian_data['sentiment'] == 'neutral']['stopword'])

# Menampilkan dataframe untuk setiap kategori sentimen bahasa Indonesia
st.write("### DataFrame untuk Sentimen Positif (Bahasa Indonesia)")
st.dataframe(indonesian_data[indonesian_data['sentiment'] == 'positive'].drop(columns=['Clean Text'], errors='ignore'))
plot_wordcloud(positive_text_id, 'positive_sentiment_id.png')
st.image('positive_sentiment_id.png', use_column_width=True)

st.write("### DataFrame untuk Sentimen Negatif (Bahasa Indonesia)")
st.dataframe(indonesian_data[indonesian_data['sentiment'] == 'negative'].drop(columns=['Clean Text'], errors='ignore'))
plot_wordcloud(negative_text_id, 'negative_sentiment_id.png')
st.image('negative_sentiment_id.png', use_column_width=True)

st.write("### DataFrame untuk Sentimen Netral (Bahasa Indonesia)")
st.dataframe(indonesian_data[indonesian_data['sentiment'] == 'neutral'].drop(columns=['Clean Text'], errors='ignore'))
plot_wordcloud(neutral_text_id, 'neutral_sentiment_id.png')
st.image('neutral_sentiment_id.png', use_column_width=True)

# Fungsi untuk mendapatkan unigram yang paling sering muncul
def get_top_n_words(text_series, n):
    words = ' '.join(text_series).split()
    top_n = Counter(words).most_common(n)
    return top_n

# Mengambil 20 unigram teratas untuk masing-masing sentimen
pos_unigrams = get_top_n_words(data[data['sentiment'] == 'positive']['stopword'], 20)
neg_unigrams = get_top_n_words(data[data['sentiment'] == 'negative']['stopword'], 20)
neutral_unigrams = get_top_n_words(data[data['sentiment'] == 'neutral']['stopword'], 20)

# Dataframe untuk unigrams positif
df1 = pd.DataFrame(pos_unigrams, columns=['Teks', 'Jumlah'])

# Dataframe untuk unigrams negatif
df2 = pd.DataFrame(neg_unigrams, columns=['Teks', 'Jumlah'])

# Dataframe untuk unigrams netral
df3 = pd.DataFrame(neutral_unigrams, columns=['Teks', 'Jumlah'])

# Plot unigram untuk sentimen positif
fig1 = px.bar(df1, x='Jumlah', y='Teks', orientation='h',
              title='20 Unigram Teratas dalam Teks Positif', 
              labels={'Jumlah': 'Jumlah', 'Teks': 'Unigram'}, color='Teks')
st.plotly_chart(fig1, use_container_width=True)

# Plot unigram untuk sentimen negatif
fig2 = px.bar(df2, x='Jumlah', y='Teks', orientation='h',
              title='20 Unigram Teratas dalam Teks Negatif', 
              labels={'Jumlah': 'Jumlah', 'Teks': 'Unigram'}, color='Teks')
st.plotly_chart(fig2, use_container_width=True)

# Plot unigram untuk sentimen netral
fig3 = px.bar(df3, x='Jumlah', y='Teks', orientation='h',
              title='20 Unigram Teratas dalam Teks Netral', 
              labels={'Jumlah': 'Jumlah', 'Teks': 'Unigram'}, color='Teks')
st.plotly_chart(fig3, use_container_width=True)

def get_top_n_gram(corpus, ngram_range, n=None):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Mengambil 20 bigram teratas untuk masing-masing sentimen
pos_bigrams = get_top_n_gram(data[data['sentiment'] == 'positive']['stopword'], (2,2), 20)
neg_bigrams = get_top_n_gram(data[data['sentiment'] == 'negative']['stopword'], (2,2), 20)
neutral_bigrams = get_top_n_gram(data[data['sentiment'] == 'neutral']['stopword'], (2,2), 20)

# Dataframe untuk bigram positif
df1 = pd.DataFrame(pos_bigrams, columns=['Teks', 'Jumlah'])

# Dataframe untuk bigram negatif
df2 = pd.DataFrame(neg_bigrams, columns=['Teks', 'Jumlah'])

# Dataframe untuk bigram netral
df3 = pd.DataFrame(neutral_bigrams, columns=['Teks', 'Jumlah'])

# Plot bigram untuk sentimen positif
fig1 = px.bar(df1, x='Jumlah', y='Teks', orientation='h',
              title='20 Bigram Teratas dalam Teks Positif',
              labels={'Jumlah': 'Jumlah', 'Teks': 'Bigram'}, color='Teks')
st.plotly_chart(fig1, use_container_width=True)

# Plot bigram untuk sentimen negatif
fig2 = px.bar(df2, x='Jumlah', y='Teks', orientation='h',
              title='20 Bigram Teratas dalam Teks Negatif',
              labels={'Jumlah': 'Jumlah', 'Teks': 'Bigram'}, color='Teks')
st.plotly_chart(fig2, use_container_width=True)

# Plot bigram untuk sentimen netral
fig3 = px.bar(df3, x='Jumlah', y='Teks', orientation='h',
              title='20 Bigram Teratas dalam Teks Netral',
              labels={'Jumlah': 'Jumlah', 'Teks': 'Bigram'}, color='Teks')
st.plotly_chart(fig3, use_container_width=True)

# Mengambil 20 trigram teratas untuk masing-masing sentimen
pos_trigrams = get_top_n_gram(data[data['sentiment'] == 'positive']['stopword'], (3,3), 20)
neg_trigrams = get_top_n_gram(data[data['sentiment'] == 'negative']['stopword'], (3,3), 20)
neutral_trigrams = get_top_n_gram(data[data['sentiment'] == 'neutral']['stopword'], (3,3), 20)

# Dataframe untuk trigram positif
df1 = pd.DataFrame(pos_trigrams, columns=['Teks', 'Jumlah'])

# Dataframe untuk trigram negatif
df2 = pd.DataFrame(neg_trigrams, columns=['Teks', 'Jumlah'])

# Dataframe untuk trigram netral
df3 = pd.DataFrame(neutral_trigrams, columns=['Teks', 'Jumlah'])

# Plot trigram untuk sentimen positif
fig1 = px.bar(df1, x='Jumlah', y='Teks', orientation='h',
              title='20 Trigram Teratas dalam Teks Positif',
              labels={'Jumlah': 'Jumlah', 'Teks': 'Trigram'}, color='Teks')
st.plotly_chart(fig1, use_container_width=True)

# Plot trigram untuk sentimen negatif
fig2 = px.bar(df2, x='Jumlah', y='Teks', orientation='h',
              title='20 Trigram Teratas dalam Teks Negatif',
              labels={'Jumlah': 'Jumlah', 'Teks': 'Trigram'}, color='Teks')
st.plotly_chart(fig2, use_container_width=True)

# Plot trigram untuk sentimen netral
fig3 = px.bar(df3, x='Jumlah', y='Teks', orientation='h',
              title='20 Trigram Teratas dalam Teks Netral',
              labels={'Jumlah': 'Jumlah', 'Teks': 'Trigram'}, color='Teks')
st.plotly_chart(fig3, use_container_width=True)
