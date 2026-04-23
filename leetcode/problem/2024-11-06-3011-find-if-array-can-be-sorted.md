---
layout: leetcode-entry
title: "3011. Find if Array Can Be Sorted"
permalink: "/leetcode/problem/2024-11-06-3011-find-if-array-can-be-sorted/"
leetcode_ui: true
entry_slug: "2024-11-06-3011-find-if-array-can-be-sorted"
---

[3011. Find if Array Can Be Sorted](https://leetcode.com/problems/find-if-array-can-be-sorted/description/) medium
[blog post](https://leetcode.com/problems/find-if-array-can-be-sorted/solutions/6014576/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06112024-3011-find-if-array-can-be?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jrefxBfNohY)
[deep-dive](https://notebooklm.google.com/notebook/affe4998-8f56-4145-bb8b-c6381fc37b1a/audio)
![1.webp](/assets/leetcode_daily_images/5a098ede.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/792

#### Problem TLDR

Can array be sorted by adjacent swap same-1-bits-nums #medium

#### Intuition

Pay attention to the `adjacent` requirement, as it simplifies the problem: split nums by chunks and check for overlaps. (it's note to myself as I spent time in the wrong direction)

The follow up of this problem would be removing the `adjacent` rule, now it becomes interesting:

```j

    //          b
    // 0001  1  1
    // 0010  2  1
    // 0011  3  2
    // 0100  4  1
    // 0101  5  2

    // 42513
    // 11212   1: 1,2,4  2: 3,5
    //
    // 1     take smallest from `1`-b busket
    //  2
    //   3
    //    4
    //     5
    // adjucent!! < -- this is a different problem

```
We would have at most 8 buckets of the sorted numbers that we can hold in a PriorityQueue, for example:

```kotlin

        val g = nums.groupBy { it.countOneBits() }
            .mapValues { PriorityQueue(it.value) }
        var prev = 0
        return nums.none { n ->
            val n = g[n.countOneBits()]!!.poll()
            n < prev.also { prev = n }
        }

```

#### Approach

* read the description carefully
* sometimes the problem size (just 100 elements) didn't hint about the actual solution

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun canSortArray(nums: IntArray): Boolean {
        var prevMax = 0; var b = 0; var max = 0
        return nums.none { n ->
            val bits = n.countOneBits()
            if (bits != b) { prevMax = max; b = bits }
            max = max(max, n)
            n < prevMax
        }
    }

```
```rust

    pub fn can_sort_array(nums: Vec<i32>) -> bool {
        nums.chunk_by(|a, b| a.count_ones() == b.count_ones())
        .map(|c|(c.iter().min(), c.iter().max()))
        .collect::<Vec<_>>()
        .windows(2).all(|w| w[0].1 < w[1].0)
    }

```
```c++

    bool canSortArray(vector<int>& nums) {
        int bp = 0, mp = 0, m = 0;
        return none_of(begin(nums), end(nums), [&](int x) {
            int b = __builtin_popcount(x);
            if (b != bp) mp = m, bp = b;
            m = max(m, x);
            return x < mp;
        });
    }

```

