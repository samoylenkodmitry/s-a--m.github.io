---
layout: leetcode-entry
title: "944. Delete Columns to Make Sorted"
permalink: "/leetcode/problem/2025-12-20-944-delete-columns-to-make-sorted/"
leetcode_ui: true
entry_slug: "2025-12-20-944-delete-columns-to-make-sorted"
---

[944. Delete Columns to Make Sorted](https://leetcode.com/problems/delete-columns-to-make-sorted/description/) easy
[blog post](https://leetcode.com/problems/delete-columns-to-make-sorted/solutions/7425776/kotlin-rust-by-samoylenkodmitry-rg1z/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20122025-944-delete-columns-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vYRmx1MBQHk)

![67d1f3d5-cce9-4f1d-8679-a72de527fec3 (1).webp](/assets/leetcode_daily_images/681669b5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1210

#### Problem TLDR

Count unsorted columns #easy

#### Intuition

Compare column with its sorted variant.

#### Approach

* or compare adjucent rows in a column

#### Complexity

- Time complexity:
$$O(nm)$$ or nmlogn

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin
// 84ms
    fun minDeletionSize(s: Array<String>) =
        s[0].indices.map{i -> s.map{it[i]}}.count { it != it.sorted() }
```
```rust
// 1ms
    pub fn min_deletion_size(s: Vec<String>) -> i32 {
        (0..s[0].len()).filter(|&i| (1..s.len()).any(|j| s[j-1].as_bytes()[i]>s[j].as_bytes()[i])).count() as _
```

