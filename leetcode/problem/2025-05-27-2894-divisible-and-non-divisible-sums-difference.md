---
layout: leetcode-entry
title: "2894. Divisible and Non-divisible Sums Difference"
permalink: "/leetcode/problem/2025-05-27-2894-divisible-and-non-divisible-sums-difference/"
leetcode_ui: true
entry_slug: "2025-05-27-2894-divisible-and-non-divisible-sums-difference"
---

[2894. Divisible and Non-divisible Sums Difference](https://leetcode.com/problems/divisible-and-non-divisible-sums-difference/description/) easy
[blog post](https://leetcode.com/problems/divisible-and-non-divisible-sums-difference/solutions/6785229/kotlin-rust-by-samoylenkodmitry-gurb/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27052025-2894-divisible-and-non-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_kw_Q0JEyTo)
![1.webp](/assets/leetcode_daily_images/9705d2c8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1001

#### Problem TLDR

Sum non-divisible minus sum divisible #easy #math

#### Intuition

I was happy to spot we can do this in a single iteration, `sum += x % m ? x : -x`

There is an arithmetic math solution however:

```j

    // a + b = n*(n+1)/2
    // b = m * k*(k+1)/2, k = n/m, m, 2m, 3m...km
    // a - b = (a + b) - 2b
    // a - b = n*(n+1)/2 - 2*m*k*(k+1)/2

```

#### Approach

* how short can it be?

#### Complexity

- Time complexity:
$$O(n)$$, or O(1)

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 5ms (52 symbols)
    fun differenceOfSums(n: Int, m: Int) =
        (1..n).sumOf { if (it % m < 1) -it else it }

```
```kotlin

// 0ms (49 symbols)
    fun differenceOfSums(n: Int, m: Int) =
        n * (n + 1) / 2 - n / m * (n / m + 1) * m

```
```kotlin

// 13ms (47 symbols)
    fun differenceOfSums(n: Int, m: Int) =
        (1..n).sum() - n / m * (n / m + 1) * m

```
```kotlin

// 12ms (46 symbols)
    fun differenceOfSums(n: Int, m: Int) =
        (1..n).sum() - (1..n/m).sum() * m * 2

```
```rust

// 0ms
    pub fn difference_of_sums(n: i32, m: i32) -> i32 {
        n * (n + 1) / 2 - n / m * (n / m + 1) * m
    }

```
```c++

// 0ms
    int differenceOfSums(int n, int m) {
        return n * (n + 1) / 2 - n / m * (n / m + 1) * m;
    }

```

