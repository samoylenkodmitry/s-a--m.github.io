---
layout: leetcode-entry
title: "2145. Count the Hidden Sequences"
permalink: "/leetcode/problem/2025-04-21-2145-count-the-hidden-sequences/"
leetcode_ui: true
entry_slug: "2025-04-21-2145-count-the-hidden-sequences"
---

[2145. Count the Hidden Sequences](https://leetcode.com/problems/count-the-hidden-sequences/description/) medium
[blog post](https://leetcode.com/problems/count-the-hidden-sequences/solutions/6673194/kotlin-rust-by-samoylenkodmitry-y7uh/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21042025-2145-count-the-hidden-sequences?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/h0wJGlvONCU)
![1.webp](/assets/leetcode_daily_images/9d42e090.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/965

#### Problem TLDR

Possible arrays from diff array and lo..hi #medium

#### Intuition

Let's observe the numbers, `x` is some starting point:

```j

    // 3,-4,5,1,-2       -4..5   len1 = 5-(-4) = 9
    // x           x
    // x+3         x+3
    // x+3-4       x-1   -1
    // x+3-4+5     x+4
    // x+3-4+5+1   x+5    5
    // x+3-4+5+1-2 x+3
    // -1..5    in -4..5 = (-4..2, -3..3, -2..4, -1..5)
    // len2 = 5-(-1) = 6
    // len1 - len2 + 1

    // 1 -3 4        1..6, len1=6-1=5
    // x       x
    // x+1     x+1
    // x+1-3   x-2
    // x+1-3+4 x+2
    // -2..2, len2 = 2-(-2)=4
    // len1 - len2 + 1 = 5-4+1=2

    // -40            -46..53, len1=99
    // x
    // x-40
    // len2=0-(-40)=40
    // 99-40+1 = 60

```

* compute the `x_max` and `x_min`
* find how many ranges of `x_min..x_max` in the range `lo..hi`

#### Approach

* beware of the int overflow
* early exit is possible

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin [Kotlin(5ms]

    fun numberOfArrays(diff: IntArray, lo: Int, up: Int): Int {
        var x = 0L; var a = 0L; var b = 0L
        for (d in diff) { x += d; a = max(a, x); b = min(b, x) }
        return max(0, 1L * up - lo - a + b + 1).toInt()
    }

```
```kotlin [2ms)]

    fun numberOfArrays(diff: IntArray, lo: Int, up: Int): Int {
        var x = 0; var a = 0; var b = 0; val r = up - lo
        for (d in diff) {
            x += d
            if (x > a) { a = x; if (a - b > r) return 0 }
            else if (x < b) { b = x; if (a - b > r) return 0 }
        }
        return r - a + b + 1
    }

```
```rust [Rust(0ms]

    pub fn number_of_arrays(diff: Vec<i32>, lo: i32, up: i32) -> i32 {
        let (mut x, mut a, mut b) = (0, 0, 0);
        for d in diff { x += d as i64; a = a.max(x); b = b.min(x) }
        0.max(up - lo + 1 - (a - b) as i32)
    }

```
```rust [0ms)]

    pub fn number_of_arrays(diff: Vec<i32>, lo: i32, up: i32) -> i32 {
        let (mut x, mut a, mut b) = (0, 0, 0);
        for d in diff { x += d; a = a.max(x); b = b.min(x);
            if a - b > up - lo { return 0 } }
        up - lo + 1 - a + b
    }

```
```c++ [C++(0ms)]

    int numberOfArrays(vector<int>& diff, int lo, int up) {
        long long x = 0, a = 0, b = 0;
        for (int d: diff) a = max(a, x += d), b = min(b, x);
        return (int) max(0LL, b - a + up - lo + 1);
    }

```

