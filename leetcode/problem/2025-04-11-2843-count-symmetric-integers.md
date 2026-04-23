---
layout: leetcode-entry
title: "2843. Count Symmetric Integers"
permalink: "/leetcode/problem/2025-04-11-2843-count-symmetric-integers/"
leetcode_ui: true
entry_slug: "2025-04-11-2843-count-symmetric-integers"
---

[2843. Count Symmetric Integers](https://leetcode.com/problems/count-symmetric-integers/description/) easy
[blog post](https://leetcode.com/problems/count-symmetric-integers/solutions/6639045/kotlin-rust-by-samoylenkodmitry-zi6s/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11042025-2843-count-symmetric-integers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/r63yRhtbxJw)
![1.webp](/assets/leetcode_daily_images/9ee74331.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/955

#### Problem TLDR

Numbers with equal half' sums #easy

#### Intuition

Brute. Force.

#### Approach

* convert to string and don't

#### Complexity

- Time complexity:
$$O((h - l)lg(h))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countSymmetricIntegers(low: Int, high: Int) =
        (low..high).count {
            val s =  "$it".map { it - '0' }; val n = s.size
            n % 2 < 1 && s.sum() == s.take(n / 2).sum() * 2
        }

```
```rust

    pub fn count_symmetric_integers(low: i32, high: i32) -> i32 {
        (low..=high).filter(|&x| {
            let (mut a, mut b, mut s, mut h, mut c) = (x, x, 0, 0, 0);
            while a > 0  { s += a % 10; a /= 10; c += 1;
                         if c % 2 > 0 { h += b % 10; b /= 10 }}
            c & 1 < 1 && s == h * 2
        }).count() as _
    }

```
```c++

    int countSymmetricIntegers(int low, int high) {
        int r = 0;
        for (int x = low; x <= high; ++x) {
            int a = x, b = x, s = 0, h = 0, c = 0;
            while (a) {
                s += a % 10; a /= 10; ++c;
                if (c % 2) { h += b % 10; b /= 10; }
            }
            r += c % 2 < 1 && s == h * 2;
        }
        return r;
    }

```

