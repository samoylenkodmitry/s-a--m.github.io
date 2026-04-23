---
layout: leetcode-entry
title: "3379. Transformed Array"
permalink: "/leetcode/problem/2026-02-05-3379-transformed-array/"
leetcode_ui: true
entry_slug: "2026-02-05-3379-transformed-array"
---

[3379. Transformed Array](https://leetcode.com/problems/transformed-array/description/) easy
[blog post](https://leetcode.com/problems/transformed-array/solutions/7554382/kotlin-rust-by-samoylenkodmitry-yka5/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05022026-3379-transformed-array?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/hybx24nUqmg)

![4e8b7ea9-79b8-45a4-b35d-bb8e83d8dfee (1).webp](/assets/leetcode_daily_images/1b403862.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1259

#### Problem TLDR

Shuffled list i+n[i] #easy

#### Intuition

It took me 8 minutes to understand the problem.
i - is the index of a result array
i+n[i] is the index of a value we take

#### Approach

* in Kotlin & Rust we have built-in for negative mod: `mod` & `rem_euclid`
* in-place solution possible if we store results in a left bits

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 162ms
    fun constructTransformedArray(n: IntArray) =
    List(n.size) { n[(it + n[it]).mod(n.size)] }
```
```rust
// 3ms
    pub fn construct_transformed_array(n: Vec<i32>) -> Vec<i32> {
        (0..n.len()).map(|i|n[((n[i]+i as i32).rem_euclid(n.len() as i32)) as usize]).collect()
    }
```

