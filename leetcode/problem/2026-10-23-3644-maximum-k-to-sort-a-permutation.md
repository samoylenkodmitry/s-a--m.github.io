---
layout: leetcode-entry
title: "3644. Maximum K to Sort a Permutation"
permalink: "/leetcode/problem/2026-10-23-3644-maximum-k-to-sort-a-permutation/"
leetcode_ui: true
entry_slug: "2026-10-23-3644-maximum-k-to-sort-a-permutation"
---

[3644. Maximum K to Sort a Permutation](https://leetcode.com/problems/maximum-k-to-sort-a-permutation/solutions/8584773/kotlin-rust-by-samoylenkodmitry-g7r7/) medium
[substack](https://dmitriisamoilenko.substack.com/p/23102026-3644-maximum-k-to-sort-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nv5kyYQd-N8)

https://dmitrysamoylenko.com/leetcode/

![23.10.2026.webp](/assets/leetcode_daily_images/23.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1520

#### Problem TLDR

Max K to sort by swapping a&b=k

#### Intuition

Each unsorted position forms a cycle island. Walk the cycle and do the 'and' operation. Then join the islands with 'and' operation.

#### Approach

* we don't have to walk islands, it is just all unsorted positions's 'and' operation

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun sortPermutation(n: IntArray) =
    n.filterIndexed{i,v->v!=i}.reduceOrNull(Int::and)?:0
```
```rust
    pub fn sort_permutation(n: Vec<i32>) -> i32 {
        let mut r=-1;for(i,v)in(0..).zip(n){if v!=i{r&=v}}r.max(0)
    }
```

