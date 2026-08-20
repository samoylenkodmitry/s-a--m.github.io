---
layout: leetcode-entry
title: "3069. Distribute Elements Into Two Arrays I"
permalink: "/leetcode/problem/2026-08-20-3069-distribute-elements-into-two-arrays-i/"
leetcode_ui: true
entry_slug: "2026-08-20-3069-distribute-elements-into-two-arrays-i"
---

[3069. Distribute Elements Into Two Arrays I](https://leetcode.com/problems/distribute-elements-into-two-arrays-i/solutions/8471812/kotlin-rust-by-samoylenkodmitry-a9gf/) easy
[substack](https://dmitriisamoilenko.substack.com/p/20082026-3069-distribute-elements?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pbKYTloQhRg)

https://dmitrysamoylenko.com/leetcode/

![20.08.2026.webp](/assets/leetcode_daily_images/20.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1457

#### Problem TLDR

Shuffle by adding to the biggest half

#### Intuition

Just follow the description

#### Approach

* optimized version uses no extra containers
* can be done in-place?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun resultArray(n: IntArray) = run {
        val a = arrayListOf(n[0]); val b = arrayListOf(n[1])
        for (x in n.drop(2)) (if (a.last() > b.last()) a else b) += x
        a + b
    }
```
```rust
    pub fn result_array(n: Vec<i32>) -> Vec<i32> {
        let (mut a, mut b) = (vec![n[0]], vec![n[1]]);
        for &x in &n[2..] {
            (if a.last() > b.last() { &mut a } else { &mut b }).push(x);
        }
        a.extend(b); a
    }
```

