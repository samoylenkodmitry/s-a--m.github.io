---
layout: leetcode-entry
title: "1295. Find Numbers with Even Number of Digits"
permalink: "/leetcode/problem/2025-04-30-1295-find-numbers-with-even-number-of-digits/"
leetcode_ui: true
entry_slug: "2025-04-30-1295-find-numbers-with-even-number-of-digits"
---

[1295. Find Numbers with Even Number of Digits](https://leetcode.com/problems/find-numbers-with-even-number-of-digits/description/) easy
[blog post](https://leetcode.com/problems/find-numbers-with-even-number-of-digits/solutions/6701042/kotlin-rust-by-samoylenkodmitry-ucch/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30042025-1295-find-numbers-with-even?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZvBKPB7VZAs)
![1.webp](/assets/leetcode_daily_images/88847816.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/974

#### Problem TLDR

Even length numbers #easy

#### Intuition

Do what is asked

#### Approach

* some golf and counter acrobatics possible
* how the input range can be used?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 3ms
    fun findNumbers(n: IntArray) =
    n.count { "$it".length % 2 < 1 }

```
```kotlin

// 1ms
    fun findNumbers(n: IntArray) = n
    .count { it in 10..99 || it in 1000..9999 || it == 100000 }

```
```rust

// 0ms
    pub fn find_numbers(n: Vec<i32>) -> i32 {
        n.iter().map(|&x| { let (mut x, mut c) = (x, 1);
            while x > 0 { x /= 10; c = 1 - c }; c
        }).sum()
    }

```
```c++

// 0ms
    int findNumbers(vector<int>& n) {
        int r = 0;
        for (int c = 0; int x: n)
            for (c = 1, r++; x > 0; x /= 10 ) r +=  2 * (++c & 1) - 1;
        return r;
    }

```

