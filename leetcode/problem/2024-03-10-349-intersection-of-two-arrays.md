---
layout: leetcode-entry
title: "349. Intersection of Two Arrays"
permalink: "/leetcode/problem/2024-03-10-349-intersection-of-two-arrays/"
leetcode_ui: true
entry_slug: "2024-03-10-349-intersection-of-two-arrays"
---

[349. Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/description/) easy
[blog post](https://leetcode.com/problems/intersection-of-two-arrays/solutions/4852330/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10032024-349-intersection-of-two?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9_0lZiioirw)
![image.png](/assets/leetcode_daily_images/edc2fc98.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/534

#### Problem TLDR

Intersection of two nums arrays #easy

#### Intuition

Built-in `Set` has an `intersect` method, that will do the trick. However, as a follow up, there is a O(1) memory solution using sorting (can be done with O(1) memory https://stackoverflow.com/questions/55008384/can-quicksort-be-implemented-in-c-without-stack-and-recursion), then just use two-pointers pattern, move the lowest:

```rust
...
      if nums1[i] < nums2[j] { i += 1 } else
      if nums1[i] > nums2[j] { j += 1 } else {
        let x = nums1[i]; res.push(x);
        while (i < nums1.len() && nums1[i] == x) { i += 1 }
        while (j < nums2.len() && nums2[j] == x) { j += 1 }
      }
...
```

#### Approach

Let's write shorter code, to save our own space and time by using built-in implementations.
* Rust wants `into_iter` instead of `iter`, as `iter` makes `vec<&&i32>`
* Rust didn't compile without `cloned`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun intersection(nums1: IntArray, nums2: IntArray) =
    nums1.toSet().intersect(nums2.toSet()).toIntArray()

```
```rust

  pub fn intersection(mut nums1: Vec<i32>, mut nums2: Vec<i32>) -> Vec<i32> {
    nums1.into_iter().collect::<HashSet<_>>()
    .intersection(&nums2.into_iter().collect()).cloned().collect()
  }

```

