---
layout: leetcode-entry
title: "3147. Taking Maximum Energy From the Mystic Dungeon"
permalink: "/leetcode/problem/2025-10-10-3147-taking-maximum-energy-from-the-mystic-dungeon/"
leetcode_ui: true
entry_slug: "2025-10-10-3147-taking-maximum-energy-from-the-mystic-dungeon"
---

[3147. Taking Maximum Energy From the Mystic Dungeon](https://leetcode.com/problems/taking-maximum-energy-from-the-mystic-dungeon/) medium
[blog post](https://leetcode.com/problems/taking-maximum-energy-from-the-mystic-dungeon/solutions/7263797/kotlin-rust-by-samoylenkodmitry-wtmn/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10102025-3147-taking-maximum-energy?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RAU7BBuYb98)

![6c72b184-248f-4f18-8f48-de7d078824ff (1).webp](/assets/leetcode_daily_images/6afe50d0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1138

#### Problem TLDR

Max k-distant suffix sum #medium #array

#### Intuition

Each `k` sums have stable positions. Track `k` sums, drop if negative.

#### Approach

* can be done in-place

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(1) with input mutation

#### Code

```kotlin

// 724ms
    fun maximumEnergy(e: IntArray, k: Int) = (e.size-1 downTo 0)
    .maxOf { e[it] += if (it+k < e.size) e[it+k] else 0; e[it] }

```
```rust

// 21ms
    pub fn maximum_energy(mut e: Vec<i32>, k: i32) -> i32 {
        let k = k as usize; for i in k..e.len() { e[i%k] = e[i] + 0.max(e[i%k]) }
        *e[..k].iter().max().unwrap()
    }

```

