---
layout: leetcode-entry
title: "1636. Sort Array by Increasing Frequency"
permalink: "/leetcode/problem/2024-07-23-1636-sort-array-by-increasing-frequency/"
leetcode_ui: true
entry_slug: "2024-07-23-1636-sort-array-by-increasing-frequency"
---

[1636. Sort Array by Increasing Frequency](https://leetcode.com/problems/sort-array-by-increasing-frequency/description/) easy
[blog post](https://leetcode.com/problems/sort-array-by-increasing-frequency/solutions/5520670/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23072024-1636-sort-array-by-increasing?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZVR0kndyTnY)
![2024-07-23_08-14.webp](/assets/leetcode_daily_images/ce9bd5be.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/679

#### Problem TLDR

Sort by frequency or descending #easy

#### Intuition

Sort with comparator.
Another way is to do sorting two times but with a stable sort (in Kotlin it is by default, in Rust you must use sort instead of sort_unstable).

#### Approach

* pay attention: there are negative numbers
* Kotlin doesn't have sortWith for IntArray

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun frequencySort(nums: IntArray): IntArray {
        val f = IntArray(202)
        for (n in nums) f[n + 100]++
        return nums
            .sortedWith(compareBy({ f[it + 100]}, { -it }))
            .toIntArray()
    }

```
```rust

    pub fn frequency_sort(mut nums: Vec<i32>) -> Vec<i32> {
        let mut f = vec![0; 201];
        for n in &nums { f[(n + 100) as usize] += 1 }
        nums.sort_unstable_by_key(|n| (f[(n + 100) as usize], -n));
        nums
    }

```

