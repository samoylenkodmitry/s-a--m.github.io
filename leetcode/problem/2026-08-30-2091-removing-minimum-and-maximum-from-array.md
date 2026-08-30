---
layout: leetcode-entry
title: "2091. Removing Minimum and Maximum From Array"
permalink: "/leetcode/problem/2026-08-30-2091-removing-minimum-and-maximum-from-array/"
leetcode_ui: true
entry_slug: "2026-08-30-2091-removing-minimum-and-maximum-from-array"
---

[2091. Removing Minimum and Maximum From Array](https://leetcode.com/problems/removing-minimum-and-maximum-from-array/solutions/8490414/kotlin-rust-by-samoylenkodmitry-guj3/) medium
[substack](https://dmitriisamoilenko.substack.com/p/30082026-2091-removing-minimum-and?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/TGWDMzAp9K4)

https://dmitrysamoylenko.com/leetcode/

![30.08.2026.webp](/assets/leetcode_daily_images/30.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1467

#### Problem TLDR

Remove min and max by cutting the tails

#### Intuition

Either remove suffix of both, or prefix of both, or each own tail suffix and prefix.

#### Approach

* shortest varian by using 'sorted'

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun minimumDeletions(n: IntArray) = n.run {
        val (a, b) = listOf(indexOf(min()), indexOf(max())).sorted()
        minOf(b + 1, size - a, a + 1 + size - b)
    }
```
```rust
    pub fn minimum_deletions(n: Vec<i32>) -> i32 {
        let (i, j) = (n.iter().position_min().unwrap(), n.iter().position_max().unwrap());
        (i.max(j) + 1).min(n.len() - i.min(j)).min(n.len() + 1 - i.abs_diff(j)) as _
    }
```

