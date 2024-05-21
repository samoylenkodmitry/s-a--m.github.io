---
layout: post
title: How to Iterate Through Arrays Faster
---

# Trust No One, Always Verify Yourself

To compare, let's write several versions of iteration. To prevent the compiler from discarding useless code, we'll add a sum of numbers calculation:

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

We'll compile with the following versions:


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

Let's examine the bytecode from the resulting apk:


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

We'll measure "by eye" (using SystemClock.elapsedRealtimeNanos) on a OnePlus 3T device (Android 8, API 26, Snapdragon 835, 6Gb).
A separate array of size 100,000,000 is allocated for each run. The action takes place in the main thread.
We'll run it 10 times, looking at the average, minimum, maximum, and total time. (nanoseconds)

ox | avg | min | max | count | sum
--- | --- | --- | --- | --- | ---
o0 | 188098202 | 141603698 | 531154323 | 10 | 1880982029
o1 | 169045885 | 133646093 | 469794063 | 10 | 1690458854
o2 | 223643875 | 174188073 | 631305521 | 10 | 2236438751
o3 | 220880307 | 173801302 | 596273073 | 10 | 2208803071
o4 | 230323713 | 175824167 | 679445365 | 10 | 2303237132
o5 | 220798156 | 173832551 | 550448177 | 10 | 2207981563


Winner - o1, not caching the array size.
Close behind it is o0 using enhanced-loop notation.

o5 and o3 are approximately equal.
In last places are o2 and o4.

Unexpectedly.

Let's try on Megafon Login (Android 4.4.2, MediaTek MT6582, 1Gb). Failed to create an array of 100 million ints due to OOM, so only 1 million.

ox | avg | min | max | count | sum
--- | --- | --- | --- | --- | ---
o0 | 57187707 | 53497769 | 62039923 | 10 | 571877076
o1 | 136767115 | 126513538 | 176206077 | 10 | 1367671153
o2 | 59381607 | 53819615 | 81372308 | 10 | 593816078
o3 | 58443446 | 54400077 | 66102384 | 10 | 584434461
o4 | 59261446 | 53455846 | 73940461 | 10 | 592614463
o5 | 60955246 | 53510924 | 80854770 | 10 | 609552461

A different scenario:
The leader remains o0 with the enhanced-loop.
On the same level with it are o3, o4, o2, and o5.
Far behind them all is o1 - the method without caching array.length.

Finally, the 2018 flagship - Pixel XL (Android P, API 29). Array size is 1 million.

ox | avg | min | max | count | sum
--- | --- | --- | --- | --- | ---
o0 | 24985637 | 16415261 | 99575687 | 10 | 249856378
o1 | 26105393 | 16737970 | 107695114 | 10 | 261053932
o2 | 33134024 | 17729273 | 104561312 | 10 | 331340242
o3 | 81853331 | 41240004 | 133074909 | 10 | 818533314
o4 | 56166797 | 47559536 | 95691885 | 10 | 561667976
o5 | 42000582 | 34823546 | 90243446 | 10 | 420005824

The leader is the enhanced-loop o0, closely followed by o1 with non-cached array.length.
Mid-range performers are o2, o5. Slightly slower is o4.
Things are quite bad for o3, going through the array in reverse order.

It seems like an optimization, as ART apparently only recognizes the standard for loop.

# Conclusions

As we can see in the bytecode, there is a difference between comparison with a variable and comparison with zero, in one case if-ge, in the other if-ltz, if-gez. But the main noticeable difference (in Android 4) is made by calling array-length.

In general, the recommendation to use enhanced-loop is valid for all supported Android APIs.

