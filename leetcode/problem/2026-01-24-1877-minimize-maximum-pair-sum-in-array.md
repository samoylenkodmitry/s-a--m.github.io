---
layout: leetcode-entry
title: "1877. Minimize Maximum Pair Sum in Array"
permalink: "/leetcode/problem/2026-01-24-1877-minimize-maximum-pair-sum-in-array/"
leetcode_ui: true
entry_slug: "2026-01-24-1877-minimize-maximum-pair-sum-in-array"
---

[1877. Minimize Maximum Pair Sum in Array](https://leetcode.com/problems/minimize-maximum-pair-sum-in-array/description) medium
[blog post](https://leetcode.com/problems/minimize-maximum-pair-sum-in-array/solutions/7520290/kotlin-rust-by-samoylenkodmitry-h8ac/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24012026-1877-minimize-maximum-pair?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nzghO7COqZQ)

![394c3372-ac47-4f65-9e3c-129ebc57ea97 (1).webp](/assets/leetcode_daily_images/bd1de6a2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1247

#### Problem TLDR

Min max pair #medium #brainteaser

#### Intuition

Sort. Pair maxes with mins.

#### Approach

* counting sort works too

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 629ms
    fun minPairSum(n: IntArray) =
    n.run{sort();zip(reversed(),Int::plus).max()}
```
```rust
// 13ms
    pub fn min_pair_sum(mut n: Vec<i32>) -> i32 {
        let (mut c,mut i) = ([0; 100001],0); for &x in &n { c[x as usize]+=1 }
        for v in 0..100001 { for _ in 0..c[v] { n[i]=v as i32;i+=1}}
        n.iter().zip(n.iter().rev()).map(|(a,b)|a+b).max().unwrap()
    }
```

