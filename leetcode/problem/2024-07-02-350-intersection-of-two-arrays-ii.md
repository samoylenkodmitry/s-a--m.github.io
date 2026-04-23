---
layout: leetcode-entry
title: "350. Intersection of Two Arrays II"
permalink: "/leetcode/problem/2024-07-02-350-intersection-of-two-arrays-ii/"
leetcode_ui: true
entry_slug: "2024-07-02-350-intersection-of-two-arrays-ii"
---

[350. Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/description/) easy
[blog post](https://leetcode.com/problems/intersection-of-two-arrays-ii/solutions/5400615/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/2072024-350-intersection-of-two-arrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/qya4KzC40OQ)
![2024-07-02_07-44_1.webp](/assets/leetcode_daily_images/c8c75c2e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/657

#### Problem TLDR

Array intersection with duplicates #easy

#### Intuition

We can do sorting and two pointers.
If nums2 on a hard disk, let's not touch it, just iterate once. For nums1 we can use a counting sort for O(n) solution.
For code golf, we can modify nums1 in-place with O(n^2) solution.

#### Approach

Golf in Kotlin, can you make it shorter?
Counting sort in Rust.

#### Complexity

- Time complexity:
$$O(n)$$ for counting sort, O(nlogn) for both sort & two pointers

- Space complexity:
$$O(n)$$ for counting sort (n = 1000), O(1) for sort & two pointers

#### Code

```kotlin

    fun intersect(nums1: IntArray, nums2: IntArray) = nums2.filter {
        val i = nums1.indexOf(it); if (i >= 0) nums1[i] = -1; i >= 0
    }

```
```rust

    pub fn intersect(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        let mut f = vec![0; 1001]; for n in nums1 { f[n as usize] += 1 }
        nums2.into_iter().filter(|&n| {
            let b = f[n as usize] > 0; f[n as usize] -= 1; b
        }).collect()
    }

```

