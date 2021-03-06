---
layout: post
title: Как быстрее итерировать массивы
---
# Никому не доверяй, все сам проверяй

Для сравнения напишем несколько вариантов итерации. Чтобы компилятор не выкидывал бесполезный код, добавим подсчет суммы чисел:
```java
public class Test {
	
	public static int o0(final int[] arr) {
		int s = 0;
		for (final int i : arr) {
			s += i;
		}
		return s;
	}
	
	public static int o1(final int[] arr) {
		int s = 0;
		for (int i = 0; i < arr.length; i++) {
			s += arr[i];
		}
		return s;
	}
	
	public static int o2(final int[] arr) {
		int s = 0;
		for (int i = 0, sz = arr.length; i < sz; i++) {
			s += arr[i];
		}
		return s;
	}
	
	public static int o3(final int[] arr) {
		int s = 0;
		for (int i = arr.length - 1; i >= 0; i--) {
			s += arr[i];
		}
		return s;
	}
	
	public static int o4(final int[] arr) {
		int s = 0;
		int i = arr.length - 1;
		for (; ; ) {
			s += arr[i];
			i--;
			if (i < 0) {
				break;
			}
		}
		return s;
	}
	
	public static int o5(final int[] arr) {
		int s = 0;
		int i = arr.length - 1;
		while (i >= 0) {
			s += arr[i];
			i--;
		}
		return s;
	}
	
}
```

Скомпилируем со следующими версиями:
```
targetSdkVersion 27
minSdkVersion 16
compileSdkVersion 28
buildToolsVersion '28.0.2'
classpath 'com.android.tools.build:gradle:3.1.3'
minifyEnabled false
shrinkResources false
debuggable true
android.enableD8=true
```

Посмотрим байт-код из получившегося apk:

```java
.class public Lru/ivi/client/screensimpl/downloadscatalogserial/Test;
.super Ljava/lang/Object;
.source "Test.java"


# direct methods
.method public constructor <init>()V
    .registers 1

    .line 3
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method

.method public static o0([I)I
    .registers 5
    .param p0, "arr"    # [I

    .line 6
    const/4 v0, 0x0

    .line 7
    .local v0, "s":I
    array-length v1, p0

    const/4 v2, 0x0

    :goto_3
    if-ge v2, v1, :cond_b

    aget v3, p0, v2

    .line 8
    .local v3, "i":I
    add-int/2addr v0, v3

    .line 7
    .end local v3    # "i":I
    add-int/lit8 v2, v2, 0x1

    goto :goto_3

    .line 10
    :cond_b
    return v0
.end method

.method public static o1([I)I
    .registers 4
    .param p0, "arr"    # [I

    .line 14
    const/4 v0, 0x0

    .line 15
    .local v0, "s":I
    const/4 v1, 0x0

    .line 15
    .local v1, "i":I
    :goto_2
    array-length v2, p0

    if-ge v1, v2, :cond_b

    .line 16
    aget v2, p0, v1

    add-int/2addr v0, v2

    .line 15
    add-int/lit8 v1, v1, 0x1

    goto :goto_2

    .line 18
    .end local v1    # "i":I
    :cond_b
    return v0
.end method

.method public static o2([I)I
    .registers 5
    .param p0, "arr"    # [I

    .line 22
    const/4 v0, 0x0

    .line 23
    .local v0, "s":I
    const/4 v1, 0x0

    .line 23
    .local v1, "i":I
    array-length v2, p0

    .line 23
    .local v2, "sz":I
    :goto_3
    if-ge v1, v2, :cond_b

    .line 24
    aget v3, p0, v1

    add-int/2addr v0, v3

    .line 23
    add-int/lit8 v1, v1, 0x1

    goto :goto_3

    .line 26
    .end local v1    # "i":I
    .end local v2    # "sz":I
    :cond_b
    return v0
.end method

.method public static o3([I)I
    .registers 4
    .param p0, "arr"    # [I

    .line 30
    const/4 v0, 0x0

    .line 31
    .local v0, "s":I
    array-length v1, p0

    add-int/lit8 v1, v1, -0x1

    .line 31
    .local v1, "i":I
    :goto_4
    if-ltz v1, :cond_c

    .line 32
    aget v2, p0, v1

    add-int/2addr v0, v2

    .line 31
    add-int/lit8 v1, v1, -0x1

    goto :goto_4

    .line 34
    .end local v1    # "i":I
    :cond_c
    return v0
.end method

.method public static o4([I)I
    .registers 4
    .param p0, "arr"    # [I

    .line 38
    const/4 v0, 0x0

    .line 39
    .local v0, "s":I
    array-length v1, p0

    add-int/lit8 v1, v1, -0x1

    .line 41
    .local v1, "i":I
    :cond_4
    aget v2, p0, v1

    add-int/2addr v0, v2

    .line 42
    add-int/lit8 v1, v1, -0x1

    .line 43
    if-gez v1, :cond_4

    .line 44
    nop

    .line 47
    return v0
.end method

.method public static o5([I)I
    .registers 4
    .param p0, "arr"    # [I

    .line 51
    const/4 v0, 0x0

    .line 52
    .local v0, "s":I
    array-length v1, p0

    add-int/lit8 v1, v1, -0x1

    .line 53
    .local v1, "i":I
    :goto_4
    if-ltz v1, :cond_c

    .line 54
    aget v2, p0, v1

    add-int/2addr v0, v2

    .line 55
    add-int/lit8 v1, v1, -0x1

    goto :goto_4

    .line 57
    :cond_c
    return v0
.end method
```

Померяем "на глаз" (через SystemClock.elapsedRealtimeNanos) на устройстве OnePlus 3T (Android 8, API 26, Snapdragon 835, 6Gb).
На каждый запуск выделяется отдельный массив размером 100_000_000. Действие происходит в главном потоке.
Запустим 10 раз, смотрим на среднее, минимальное, максимальное и суммарное время. (наносекунды)

ox | avg | min | max | count | sum
--- | --- | --- | --- | --- | ---
o0 | 188098202 | 141603698 | 531154323 | 10 | 1880982029
o1 | 169045885 | 133646093 | 469794063 | 10 | 1690458854
o2 | 223643875 | 174188073 | 631305521 | 10 | 2236438751
o3 | 220880307 | 173801302 | 596273073 | 10 | 2208803071
o4 | 230323713 | 175824167 | 679445365 | 10 | 2303237132
o5 | 220798156 | 173832551 | 550448177 | 10 | 2207981563


Победитель - o1, не кеширующий размер массива.
Не сильно отстает от него o0 использующий enhanced-loop запись.

o5 и o3 приблизительно равны.
На последних местах o2 и o4.

Неожиданно.

Попробуем Megafon Login (Android 4.4.2, MediaTek MT6582, 1Gb). Создать массив из 100млн интов на нем не удалось из-за OOM, поэтому только 1млн.

ox | avg | min | max | count | sum
--- | --- | --- | --- | --- | ---
o0 | 57187707 | 53497769 | 62039923 | 10 | 571877076
o1 | 136767115 | 126513538 | 176206077 | 10 | 1367671153
o2 | 59381607 | 53819615 | 81372308 | 10 | 593816078
o3 | 58443446 | 54400077 | 66102384 | 10 | 584434461
o4 | 59261446 | 53455846 | 73940461 | 10 | 592614463
o5 | 60955246 | 53510924 | 80854770 | 10 | 609552461

Немного другой расклад:
В лидерах остался o0 enhanced-loop. 
На одном уровне с ним o3, o4, o2 и o5. 
Сильно проиграл всем o1 - метод без кеширования array.length.

Наконец, пирог 2018 - Pixel XL (Android P, API 29). Размер массива 1млн.

ox | avg | min | max | count | sum
--- | --- | --- | --- | --- | ---
o0 | 24985637 | 16415261 | 99575687 | 10 | 249856378
o1 | 26105393 | 16737970 | 107695114 | 10 | 261053932
o2 | 33134024 | 17729273 | 104561312 | 10 | 331340242
o3 | 81853331 | 41240004 | 133074909 | 10 | 818533314
o4 | 56166797 | 47559536 | 95691885 | 10 | 561667976
o5 | 42000582 | 34823546 | 90243446 | 10 | 420005824

Лидер - enhanced-loop o0, рядом с ним o1 с некешированным array.length.
Середнячки - o2, o5. Чуть медленнее o4.
Совсем плохи дела у o3, проходящего массив в обратном порядке.

Похоже на оптимизацию, видимо ART распознает только обычный цикл for.

# Выводы

Как видим в байткоде есть разница между сравнением с переменной и сравнением с нулем, в одном случае if-ge, в другом if-ltz, if-gez. Но основную заметную разницу (в Android 4) делает вызов array-length. 

В целом рекомендация использовать enhanced-loop актуальна для всех поддерживаемых android-API.

