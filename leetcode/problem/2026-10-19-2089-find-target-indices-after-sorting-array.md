---
layout: leetcode-entry
title: "2089. Find Target Indices After Sorting Array"
permalink: "/leetcode/problem/2026-10-19-2089-find-target-indices-after-sorting-array/"
leetcode_ui: true
entry_slug: "2026-10-19-2089-find-target-indices-after-sorting-array"
---

[2089. Find Target Indices After Sorting Array](https://leetcode.com/problems/find-target-indices-after-sorting-array/solutions/8578960/kotlin-rust-by-samoylenkodmitry-5chx/) easy
[substack](https://dmitriisamoilenko.substack.com/p/19102026-2089-find-target-indices?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NNkNSRNEyXg)

https://dmitrysamoylenko.com/leetcode/

![19.10.2026.webp](/assets/leetcode_daily_images/19.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1516

#### Problem TLDR

Positions of target in sorted order

#### Intuition

Sort. Then filter positions.
Or count elements less and equal to the target.

#### Approach

* Rust has a nice `positions` filter

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun targetIndices(n: IntArray, t: Int)=
    n.indices.filter{n.sorted()[it]==t}
```
```rust
    pub fn target_indices(n: Vec<i32>, t: i32) -> Vec<i32> {
        n.into_iter().sorted().positions(|x| x == t).map(|i| i as _).collect()
    }
```

