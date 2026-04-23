---
layout: leetcode-entry
title: "3396. Minimum Number of Operations to Make Elements in Array Distinct"
permalink: "/leetcode/problem/2025-04-08-3396-minimum-number-of-operations-to-make-elements-in-array-distinct/"
leetcode_ui: true
entry_slug: "2025-04-08-3396-minimum-number-of-operations-to-make-elements-in-array-distinct"
---

[3396. Minimum Number of Operations to Make Elements in Array Distinct](https://leetcode.com/problems/minimum-number-of-operations-to-make-elements-in-array-distinct/description/) easy
[blog post](https://leetcode.com/problems/minimum-number-of-operations-to-make-elements-in-array-distinct/solutions/6628052/kotlin-rust-by-samoylenkodmitry-zvvz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08042025-3396-minimum-number-of-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rjDPkSjQekA)
![1.webp](/assets/leetcode_daily_images/e280a586.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/952

#### Problem TLDR

Count removals 3-prefixes to deduplicate #easy

#### Intuition

Brute force is accepted.

Linear solution: start from the tail and stop on first duplicate.

#### Approach

Observations by golfing:

* `count` works, after some border we always have duplicate (meaning, we also can do a binary search)
* forward pass possible (and gives faster speed with CPU caches)
* bitset can be used

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(max)$$

#### Code

```kotlin

    fun minimumOperations(nums: IntArray) = nums.indices
    .count { nums.drop(it * 3).let { it.distinct() != it }}

```
```kotlin

    fun minimumOperations(nums: IntArray): Int {
        val o = IntArray(101)
        return nums.withIndex().maxOf { (i, x) ->
            o[x].also { o[x] = 1 + i / 3 }
        }
    }

```
```kotlin

    fun minimumOperations(nums: IntArray): Int {
        val f = IntArray(101)
        for (i in nums.lastIndex downTo 0)
            if (f[nums[i]]++ > 0) return 1 + i / 3
        return 0
    }

```
```rust

    pub fn minimum_operations(nums: Vec<i32>) -> i32 {
        let mut o = [0; 101];
        nums.iter().enumerate().map(|(i, &x)| { let x = x as usize;
            let r = o[x]; o[x] = 1 + i as i32 / 3; r}).max().unwrap()
    }

```
```c++

    int minimumOperations(vector<int>& nums) {
        for (int i = size(nums) - 1, f[101]; i >= 0; --i)
            if (f[nums[i]]++ > 0) return 1 + i / 3;
        return 0;
    }

```

