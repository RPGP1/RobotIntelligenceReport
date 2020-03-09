プログラム実行まで
===

## ライブラリの入手

`.gitmodule` ファイルに従って2つのライブラリ `eigen`, `cmdline` を導入する。

## ビルド

当プログラムは `cmake`, `make` を用いてビルドする。

### cmake

ルートディレクトリの `CMakeLists.txt` を参照してコンフィグする。

>   例
>   ```
>   $> mkdir build
>   $> cd build
>   $> cmake .. -DCMAKE_BUILD_TYPE=Release
>   ```

作成者の `cmake` バージョンは `3.14.0-rc2` である。

### make

`cmake` でコンフィグしてできた `Makefile` を参照してビルドする。

>   例
>   ```
>   $> make
>   ```

作成者の `make` バージョンは `GNU Make 4.1 Built for x86_64-pc-linux-gnu` である。

コンパイラは `g++ (Ubuntu 8.3.0-6ubuntu1~18.04.1) 8.3.0` である。なお、標準ライブラリ `filesystem` を用いているため、 `apt` によって `g++` にリンクされる `g++-7` ではコンパイルできないだろうと思われる。

## 実行

実行に必要な用意やオプションは、生成されるプログラムを実行した際に表示されるようになっている。

```
usage: ./bin/cnn --data=string [options] ... 
options:
  -d, --data             directory containing mnist data (string)
  -o, --output           directory to output model parameters and log (string [=result])
  -b, --batch            batch size (unsigned int [=100])
  -e, --epoch            the number of epochs (unsigned int [=100])
  -i, --save_interval    interval of saving model parameters (specified by epoch number) (unsigned int [=1])
  -m, --model            model parameters to load (specified by epoch number) (unsigned int [=0])
  -l, --learning_rate    learning rate (double [=0.001])
      --momentum_rate    momentum rate (double [=0.9])
  -n, --noise_rate       noise rate (double [=0])
  -h, --help             show this description
```

実行に必須となるのは、 `--data` で指定するMNISTデータセット(の保管場所)である。 http://yann.lecun.com/exdb/mnist/ よりダウンロードの後、適切なディレクトリで全て解凍し、そのディレクトリをこのオプションで指定する。
