---
layout: leetcode-entry
title: "3863. Minimum Operations to Sort a String"
permalink: "/leetcode/problem/2026-10-21-3863-minimum-operations-to-sort-a-string/"
leetcode_ui: true
entry_slug: "2026-10-21-3863-minimum-operations-to-sort-a-string"
---

[3863. Minimum Operations to Sort a String](https://leetcode.com/problems/minimum-operations-to-sort-a-string/solutions/8582018/kotlin-rust-by-samoylenkodmitry-vddv/) medium
[substack](https://dmitriisamoilenko.substack.com/p/21102026-3863-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/i3BMQ98Yhac)

https://dmitrysamoylenko.com/leetcode/

![21.10.2026.webp](/assets/leetcode_daily_images/21.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1518

#### Problem TLDR

Ops to sort by sorting substrings

#### Intuition

Brain teaser of corner cases.
0 if sorted
1 if first is min or last is first max
-1 if two unsorted
3 if last is min and first is last max
2 otherwise

#### Approach

* passed after 6 failed attempts

#### Complexity

- Time complexity:
$$O(nlong)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun minOperations(s: String) = s.run{when {
        toList() == toList().sorted() -> 0
        first() == min() || last() == max() -> 1
        length < 3 -> -1
        indexOf(min()) == length-1 && lastIndexOf(max()) < 1 -> 3
        else -> 2
    }}
```
```rust
    pub fn min_operations(s: String) -> i32 {
        let MinMax(min, max) = s.chars().minmax() else { return 0 };
        s.as_bytes().is_sorted().then_some(0)
            .or_else(|| (s.starts_with(min) || s.ends_with(max)).then_some(1))
            .or_else(|| (s.len() < 3).then_some(-1))
            .or_else(|| (s.rfind(max)==Some(0) && s.find(min)==Some(s.len()-1)).then_some(3))
            .unwrap_or(2)
    }
```

