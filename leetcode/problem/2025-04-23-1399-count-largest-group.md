---
layout: leetcode-entry
title: "1399. Count Largest Group"
permalink: "/leetcode/problem/2025-04-23-1399-count-largest-group/"
leetcode_ui: true
entry_slug: "2025-04-23-1399-count-largest-group"
---

[1399. Count Largest Group](https://leetcode.com/problems/count-largest-group/description/) easy
[blog post](https://leetcode.com/problems/count-largest-group/solutions/6679439/kotlin-rust-by-samoylenkodmitry-siae/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23042025-1399-count-largest-group?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/i3TsO3ruPm4)
![1.webp](/assets/leetcode_daily_images/07c8a9a6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/967

#### Problem TLDR

Count of max groups by digits sum 1..n #easy

#### Intuition

The brute-force is accepted.

#### Approach

* max digits sum is `9+9+9+9 = 36`

#### Complexity

- Time complexity:
$$O(nlg(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countLargestGroup(n: Int): Int =
        (1..n).groupBy { "$it".sumOf { it - '0' }}.values
        .run { count { it.size == maxOf { it.size }}}

```
```kotlin

    fun countLargestGroup(n: Int): Int {
        val f = IntArray(37); var cnt = 0; var gmax = 0
        for (x in 1..n) {
            var s = 0; var y = x
            while (y > 0) { s += y % 10; y /= 10 }
            val g = ++f[s]
            if (g > gmax) { gmax = g; cnt = 1 }
            else if (g == gmax) cnt++
        }
        return cnt
    }

```
```rust

    pub fn count_largest_group(n: i32) -> i32 {
        let (mut f, mut gmax, mut cnt) = ([0; 37], 0, 0);
        for x in 1..=n {
            let (mut s, mut y) = (0, x);
            while y > 0 { s += y as usize % 10; y /= 10 }
            f[s] += 1; let g = f[s];
            if g > gmax { gmax = g; cnt = 1 }
            else if g == gmax { cnt += 1 }
        } cnt
    }

```
```c++

    int countLargestGroup(int n) {
        int f[37], gmax = 0, cnt = 0;
        for (;n;n--) {
            int x = n, s = 0;
            while (x) s += x % 10, x /= 10;
            int g = ++f[s];
            if (g > gmax) gmax = g, cnt = 1;
            else if (g == gmax) cnt++;
        } return cnt;
    }

```

