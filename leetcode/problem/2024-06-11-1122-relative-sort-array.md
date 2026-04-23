---
layout: leetcode-entry
title: "1122. Relative Sort Array"
permalink: "/leetcode/problem/2024-06-11-1122-relative-sort-array/"
leetcode_ui: true
entry_slug: "2024-06-11-1122-relative-sort-array"
---

[1122. Relative Sort Array](https://leetcode.com/problems/relative-sort-array/description/) easy
[blog post](https://leetcode.com/problems/relative-sort-array/solutions/5292985/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11062024-1122-relative-sort-array?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/hBpNvGP8YYg)
![2024-06-11_07-08.webp](/assets/leetcode_daily_images/33f76331.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/636

#### Problem TLDR

Sort an array by the given order #easy

#### Intuition

Associate the `arr2`, then use it as key for sorting `arr1`.
Another solution is to use the Counting Sort: count `arr1`, then first place `arr2` values, decreasing `cnt`, and then place the remaining `cnt`.

#### Approach

* there is a `compareBy` in Kotlin that can receive several comparators
* or we can just use `n + 1001` for this problem
* notice `.cloned()` in Rust: it allows to use a value instead of pointer in `unwrap_or`

#### Complexity

- Time complexity:
$$O(nlog(n))$

- Space complexity:
$$O(m)$$

#### Code

```kotlin

    fun relativeSortArray(arr1: IntArray, arr2: IntArray) =
        arr2.withIndex().associate { (i, v) -> v to i }.let { inds ->
            arr1.sortedWith(compareBy({ inds[it] ?: 1001 }, { it }))
        }

```
```rust

    pub fn relative_sort_array(mut arr1: Vec<i32>, arr2: Vec<i32>) -> Vec<i32> {
        let mut inds = HashMap::new(); for i in 0..arr2.len() { inds.insert(arr2[i], i); }
        arr1.sort_unstable_by_key(|n| inds.get(n).cloned().unwrap_or(1001 + *n as usize));
        arr1
    }

```

