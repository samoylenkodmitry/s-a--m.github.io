---
layout: leetcode-entry
title: "1137. N-th Tribonacci Number"
permalink: "/leetcode/problem/2024-04-24-1137-n-th-tribonacci-number/"
leetcode_ui: true
entry_slug: "2024-04-24-1137-n-th-tribonacci-number"
---

[1137. N-th Tribonacci Number](https://leetcode.com/problems/n-th-tribonacci-number/description/) easy
[blog post](https://leetcode.com/problems/n-th-tribonacci-number/solutions/5065642/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24042024-1137-n-th-tribonacci-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZrwbZORpDro)
![2024-04-24_08-41.webp](/assets/leetcode_daily_images/4618915f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/582

#### Problem TLDR

`n`th Tribonacci number f(n + 3) = f(n) + f(n + 1) + f(n + 2) #easy

#### Intuition

Use tree variables and compute the result in a for-loop.

#### Approach

There are some clever approaches:
* we can use an array and loop the index
* we can try to play this with tree variables but without a temp variable

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun tribonacci(n: Int): Int {
        if (n < 2) return n
        val t = intArrayOf(0, 1, 1)
        for (i in 3..n) t[i % 3] = t.sum()
        return t[n % 3]
    }

```
```rust

    pub fn tribonacci(n: i32) -> i32 {
        if n < 2 { return n }
        let (mut t1, mut t2, mut t0t1) = (1, 1, 1);
        for _ in 2..n as usize {
            t2 += t0t1;
            t0t1 = t1 + t2 - t0t1;
            t1 = t0t1 - t1
        }; t2
    }

```

