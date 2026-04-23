---
layout: leetcode-entry
title: "3318. Find X-Sum of All K-Long Subarrays I"
permalink: "/leetcode/problem/2025-11-04-3318-find-x-sum-of-all-k-long-subarrays-i/"
leetcode_ui: true
entry_slug: "2025-11-04-3318-find-x-sum-of-all-k-long-subarrays-i"
---

[3318. Find X-Sum of All K-Long Subarrays I](https://leetcode.com/problems/find-x-sum-of-all-k-long-subarrays-i/) easy
[blog post](https://leetcode.com/problems/find-x-sum-of-all-k-long-subarrays-i/solutions/7325781/kotlin-rust-by-samoylenkodmitry-koxq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04112025-3318-find-x-sum-of-all-k?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZTmOvDEsTcs)

![d9c7334b-fc8a-482e-8ae9-8c6184d92591 (1).webp](/assets/leetcode_daily_images/14f935b8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1163

#### Problem TLDR

Sums of x most frequent from k-windows #easy

#### Intuition

Not actually easy.
Brute-force: solve for each window, count frequency map, sort, take x.

#### Approach

* only 50 numbers; use an array instead of a HashMap

#### Complexity

- Time complexity:
$$O(nklogk)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 62ms
    fun findXSum(n: IntArray, k: Int, x: Int) =
    (0..n.size-k).map { val g = n.slice(it..<it+k).groupBy {it}
        g.keys.sortedWith(compareBy({-(g[it]!!).size},{-it}))
         .take(x).map { it * g[it]!!.size }.sum()
    }

```
```rust
// 1ms
    pub fn find_x_sum(n: Vec<i32>, k: i32, x: i32) -> Vec<i32> {
        n.windows(k as usize).map(|w| {
            let mut f = [0;51]; for &v in w.iter() { f[v as usize] += 1 }
            (0..51).map(|v|(-f[v],-(v as i32))).sorted().into_iter()
            .take(x as usize).map(|(f,v)|v*f).sum()
        }).collect()
    }

```

