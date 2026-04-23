---
layout: leetcode-entry
title: "2566. Maximum Difference by Remapping a Digit"
permalink: "/leetcode/problem/2025-06-14-2566-maximum-difference-by-remapping-a-digit/"
leetcode_ui: true
entry_slug: "2025-06-14-2566-maximum-difference-by-remapping-a-digit"
---

[2566. Maximum Difference by Remapping a Digit](https://leetcode.com/problems/maximum-difference-by-remapping-a-digit/description/) easy
[blog post](https://leetcode.com/problems/maximum-difference-by-remapping-a-digit/solutions/6841863/kotlin-rust-by-samoylenkodmitry-7ham/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14062025-2566-maximum-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZdiQygc0i7s)
![1.webp](/assets/leetcode_daily_images/eed20591.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1019

#### Problem TLDR

Max - min, replacing any digit #easy

#### Intuition

Brute-force: replace any digit to any other digit, no thinking.
Brute-force2: replace any digit to '9' for max, and to '0' for min, small thinking.
Galaxy brain: replace first non-nine to nine for max, first non-zero to zero for min.

#### Approach

* we can do this in one forward pass by dividing 10^8 / 10
* instead of min and max compute diff
* we can use two arraya of transformations or just to variables to check

#### Complexity

- Time complexity:
$$O(1)$$, 10 digits, 10x10 runs, total is 1000 operations; 10 ops for optimized

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 12ms
    fun minMaxDifference(n: Int) =
        ('0'..'8').maxOf { "$n".replace(it, '9').toInt() } -
        "$n".replace("$n"[0], '0').toInt()

```
```kotlin

// 8ms
    fun minMaxDifference(n: Int) =
        "$n".replace("$n".find { it != '9'} ?: '9', '9').toInt() -
        "$n".replace("$n"[0], '0').toInt()

```
```kotlin

// 0ms
    fun minMaxDifference(num: Int): Int {
        var n = num; var pow = 100000000; var diff = 0
        while (n > 0 && n / pow == 0) pow /= 10
        var nine = 9; val zero = n / pow
        while (pow > 0) {
            val d = n / pow
            if (nine == 9 && d != 9) nine = d
            diff = diff * 10 + (if (d == nine) 9 else d) - (if (d == zero) 0 else d)
            n -= d * pow; pow /= 10
        }
        return diff
    }

```
```rust

// 0ms
    pub fn min_max_difference(mut n: i32) -> i32 {
        let mut p = 100000000; while n > 0 && n / p == 0 { p /= 10 }
        let (mut nine, zero, mut diff) = (9, n / p, 0);
        while p > 0 {
            let d = n / p; n -= d * p; p /= 10; diff *= 10;
            if nine == 9 && d != 9 { nine = d }
            diff += (if d == nine { 9 } else { d }) -
                     if d == zero { 0 } else { d }
        } diff
    }

```
```c++

// 0ms
    int minMaxDifference(int n) {
        auto s = to_string(n), t = s; int m = 0;
        for (char c = '0'; c <= '9'; ++c) {
            t = s; for (char& x: t) if (x == c) x = '9';
            m = max(m, stoi(t));
        }
        t = s; for (char& x: t) if (x == s[0]) x = '0';
        return m - stoi(t);
    }

```

