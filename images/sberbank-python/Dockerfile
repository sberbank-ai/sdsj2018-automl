FROM kaggle/python

RUN pip install tqdm pymystem3
RUN pip install dawg https://github.com/kmike/pymorphy2/archive/master.zip pymorphy2-dicts-ru
RUN pip install -U pymorphy2-dicts-ru

RUN python -c "import pymystem3.mystem ; pymystem3.mystem.autoinstall()"

RUN pip install jellyfish
