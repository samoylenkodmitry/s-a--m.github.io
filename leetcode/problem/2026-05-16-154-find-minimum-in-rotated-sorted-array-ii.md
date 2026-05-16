---
layout: leetcode-entry
title: "154. Find Minimum in Rotated Sorted Array II"
permalink: "/leetcode/problem/2026-05-16-154-find-minimum-in-rotated-sorted-array-ii/"
leetcode_ui: true
entry_slug: "2026-05-16-154-find-minimum-in-rotated-sorted-array-ii"
---

[154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/solutions/8247810/kotlin-rust-by-samoylenkodmitry-dp6l/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16052026-154-find-minimum-in-rotated?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/SmVleMrSjXI)

https://dmitrysamoylenko.com/leetcode/

![16.05.2026.webp](/assets/leetcode_daily_images/16.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1361

#### Problem TLDR

Binary search in shifted array with duplicates

#### Intuition

```j
    // 3 4 5 6 7 1 2 3
    // 3 1
    // false false
    // 3 1 3
    // false true false
```

* compare all elements with last
* slice duplicates out in O(N)

#### Approach

* use built-in functions, Rust: partition_point, position, slices [..], Kotlin: binarySearch {..}, indexOfFirst
* x.inv() is -1-x

#### Complexity

- Time complexity:
$$O(n|logn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun findMin(n: IntArray) = n[n.asList().binarySearch(
        max(0, n.indexOfFirst { it != n.last()})) {
            if (it <= n.last()) 1 else -1 }.inv()]
```
```rust
    pub fn find_min(n: Vec<i32>) -> i32 {
        let n = &n[n.iter().position(|&x| x != n[n.len()-1]).unwrap_or(0)..];
        n[n.partition_point(|&x|x > n[n.len()-1])]
    }
```

