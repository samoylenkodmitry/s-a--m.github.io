---
layout: leetcode-entry
title: "153. Find Minimum in Rotated Sorted Array"
permalink: "/leetcode/problem/2026-05-15-153-find-minimum-in-rotated-sorted-array/"
leetcode_ui: true
entry_slug: "2026-05-15-153-find-minimum-in-rotated-sorted-array"
---

[153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/solutions/8235027/kotlin-rust-by-samoylenkodmitry-9zge/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15052026-153-find-minimum-in-rotated?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/goOiR_5b-QE)

https://dmitrysamoylenko.com/leetcode/

![15.05.2026.webp](/assets/leetcode_daily_images/15.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1360

#### Problem TLDR

Binary search in shifted array

#### Intuition

```j
    // 5 6 7 0 1 2 3 4
    // l     m       h
    // 6 7 0 1 2 3 4 5
    // l     m       h
    // 6 7 0 1 2 3 4 5
    // l m   h

    // 4 5 6 7 0 1 2
    // l     m     h
    //         l
```

* invent the binary search from scratch
* or notice that we can compare all elements with last

#### Approach

* use built-in functions, Rust: partition_point, Kotlin: binarySearch {..}

#### Complexity

- Time complexity:
$$O(logn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun findMin(n: IntArray) =
        n[-1-n.asList().binarySearch { if (it > n.last()) -1 else 1}]
```
```rust
    pub fn find_min(n: Vec<i32>) -> i32 {
        n[n.partition_point(|&x|x>n[n.len()-1])]
    }
```

