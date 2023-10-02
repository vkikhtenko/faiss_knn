# faiss_knn
Проект "Мэтчинг товаров маркетплейса"
Проект содержит три файла: 
1. master_2.ipynb - основной ноутбук проекта, в которой обрабатываются данные, создается индекс faiss и проверяется метрика на валидационной выборке
2. app.py - приложение для определения пяти наиболее близких товаров из base на основе рандомногоо вектора размерностью 66
3. test.py - создает рандомный запрос для приложения app.py

Также в ходе выполнения кода в ноутбуке master_2.ipynb будет создано два файла:
5. faiss_index.pkl - файл с индексов faiss, который будет использоваться для поиска ближайших соседей в app.py
6. base_index.txt - текстовый файл со словарем {порядковый_номер: индекс_в_base}, который используется в app.py для вывода индекса товаров