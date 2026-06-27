---
layout: leetcode-entry
title: "3020. Find the Maximum Number of Elements in Subset"
permalink: "/leetcode/problem/2026-06-27-3020-find-the-maximum-number-of-elements-in-subset/"
leetcode_ui: true
entry_slug: "2026-06-27-3020-find-the-maximum-number-of-elements-in-subset"
---

[3020. Find the Maximum Number of Elements in Subset](https://leetcode.com/problems/find-the-maximum-number-of-elements-in-subset/solutions/8361248/kotlin-rust-by-samoylenkodmitry-8gdy/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27062026-3020-find-the-maximum-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/SObyptIwXkE)

https://dmitrysamoylenko.com/leetcode/

![27.06.2026.webp](/assets/leetcode_daily_images/27.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1403

#### Problem TLDR

Powers of two symmetrical exponentially growing longest sequence length

#### Intuition

```j
    // the base can be anything, not just 2
```
Compute frequencies.
The length of the sequence is very small, just check every number if it has continuation by brute force, do +2 at every step, frequencey should be at least 2.

#### Approach

* ones 1111 is the corner case, and it can be odd or even

#### Complexity

- Time complexity:
$$O(nlog(log(max)))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maximumLength(n: IntArray) = n.groupBy{1L*it}.run {
        max((get(1)?.size?:0).let{it-1+it%2},
            (keys-1).maxOfOrNull { x ->
                var (n,c) = x to 0
                while ((get(n)?.size?:0)>1) { c += 2; n *= n}
                c + if (n in this) 1 else -1
        }?:1 ) }
```
```rust
    pub fn maximum_length(n: Vec<i32>) -> i32 {
        let mut f = HashMap::new();
        for x in n { *f.entry(x as i64).or_default() += 1 }
        let o = f.remove(&1).unwrap_or(0);
        f.keys().fold(1.max(o - 1 | 1), |r, &k| {
            let (mut x, mut c) = (k, 0);
            while f.get(&x) > Some(&1) { c += 2; x *= x }
            r.max(c + (f.get(&x) > None) as i32 * 2 - 1)
        })
    }
```

