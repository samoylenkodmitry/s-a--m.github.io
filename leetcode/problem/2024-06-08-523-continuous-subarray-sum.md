---
layout: leetcode-entry
title: "523. Continuous Subarray Sum"
permalink: "/leetcode/problem/2024-06-08-523-continuous-subarray-sum/"
leetcode_ui: true
entry_slug: "2024-06-08-523-continuous-subarray-sum"
---

[523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/description/) medium
[blog post](https://leetcode.com/problems/continuous-subarray-sum/solutions/5277558/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08062024-523-continuous-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/TLT-JYx7e0A)
![2024-06-08_07-51_1.webp](/assets/leetcode_daily_images/ee4e1491.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/633

#### Problem TLDR

Any subarray sum % k = 0 #medium #hashmap

#### Intuition

Let's observe the problem examples:

```j
    // 5 0 0 0       k = 3   true?? --> [0 0] % 3 = 0
    //
    // 23   2   6   2   5    k = 8
    // 23                    23 % 8 = 0
    //     25                25 % 8 = 7
    //         31            31 % 8 = 7  (31-25)%8=31%8-25%8=0
    //             33
    //                 38
    //
    // 0 1 0 3 0 4 0 4 0  k = 7
    // 23 2   4  6  6
    // 23
    //    25
    //       29
    //          35
```

We can't just use two pointers here, because every subarray to the left can give the result in the future.
However, we can store subarray sums. But what to do with them next? If we look at example `23 2 6 2 5, k = 8`, subarray `[2 6]` is good, and it is made from sums `31` and `23`: `31 - 23 = 8` -> (31 - 23) % k = 8 % k -> 31 % k - 23 % k = k % k = 0 -> `31 % k == 23 % k`. So, our subarray `sums % k` must be equal for subarray between them be good.

The corener cases:

* For the case `5 0 0 0` result is true because there is `[0, 0]` subarray which gives `0 % k = 0`. That mean, we should store the first occurence index to check the length later.
* For the case `2 6, k = 8` we must consider entire array, so we must store the first occurence of `0` at position `-1`.

#### Approach

* `getOrPut` and `entry.or_insert` in Kotlin & Rust saves us some keystrokes

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun checkSubarraySum(nums: IntArray, k: Int): Boolean {
        val sums = HashMap<Int, Int>().apply { put(0, -1) }
        var sum = 0
        return nums.withIndex().any { (i, n) ->
            sum += n
            sums.getOrPut(sum % k) { i } < i - 1
        }
    }

```
```rust

    pub fn check_subarray_sum(nums: Vec<i32>, k: i32) -> bool {
        let (mut s, mut sums) = (0, HashMap::new()); sums.insert(0, -1);
        (0..nums.len()).any(|i| {
            s += nums[i];
            1 + *sums.entry(s % k).or_insert(i as _) < i as _
        })
    }

```

