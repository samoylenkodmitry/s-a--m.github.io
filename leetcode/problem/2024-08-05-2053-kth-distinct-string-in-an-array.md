---
layout: leetcode-entry
title: "2053. Kth Distinct String in an Array"
permalink: "/leetcode/problem/2024-08-05-2053-kth-distinct-string-in-an-array/"
leetcode_ui: true
entry_slug: "2024-08-05-2053-kth-distinct-string-in-an-array"
---

[2053. Kth Distinct String in an Array](https://leetcode.com/problems/kth-distinct-string-in-an-array/description/) easy
[blog post](https://leetcode.com/problems/kth-distinct-string-in-an-array/solutions/5588979/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05082024-2053-kth-distinct-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FAn8jqZw2B8)
![1.webp](/assets/leetcode_daily_images/81046b43.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/693

#### Problem TLDR

`kth` unique #easy

#### Intuition

Filter out all the duplicates first.

### Approach

We can use a HashMap for counter or just two HashSets.
Let's use some API:
* Kotlin: groupingBy.eachCount, filter
* Rust: filter, skip, next

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun kthDistinct(arr: Array<String>, k: Int) = arr
        .groupingBy { it }.eachCount().filter { it.value == 1 }
        .keys.elementAtOrNull(k - 1) ?: ""
```
```rust

    pub fn kth_distinct(arr: Vec<String>, k: i32) -> String {
        let (mut uniq, mut dup) = (HashSet::new(), HashSet::new());
        for s in &arr { if !uniq.insert(s) { dup.insert(s); }}
        arr.iter().filter(|&s| !dup.contains(s)).skip(k as usize - 1)
           .next().unwrap_or(&"".to_string()).to_string()
    }

```

