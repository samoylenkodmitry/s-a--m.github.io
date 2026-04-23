---
layout: leetcode-entry
title: "860. Lemonade Change"
permalink: "/leetcode/problem/2024-08-15-860-lemonade-change/"
leetcode_ui: true
entry_slug: "2024-08-15-860-lemonade-change"
---

[860. Lemonade Change](https://leetcode.com/problems/lemonade-change/description/) easy
[blog post](https://leetcode.com/problems/lemonade-change/solutions/5638776/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15082024-860-lemonade-change?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Aq1G1oqHrV0)

![1.webp](/assets/leetcode_daily_images/94260180.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/703

#### Problem TLDR

Simulate money exchange #easy #simulation

#### Intuition

* queue order must not be changed

Just simulate the process.

#### Approach

* we don't have to keep $20's

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun lemonadeChange(bills: IntArray): Boolean {
        val s = IntArray(21)
        return bills.all { b ->
            s[b]++
            if (b > 5) s[5]--
            if (b > 10) if (s[10] > 0) s[10]-- else s[5] -= 2
            s[5] >= 0
        }
    }

```
```rust

    pub fn lemonade_change(bills: Vec<i32>) -> bool {
        let (mut s5, mut s10) = (0, 0);
        bills.iter().all(|&b| {
            if b == 5 { s5 += 1 }
            if b == 10 { s10 += 1 }
            if b > 5 { s5 -= 1 }
            if b > 10 { if s10 > 0 { s10 -= 1 } else { s5 -= 2 }}
            s5 >= 0
        })
    }

```

