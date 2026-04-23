---
layout: leetcode-entry
title: "1760. Minimum Limit of Balls in a Bag"
permalink: "/leetcode/problem/2024-12-07-1760-minimum-limit-of-balls-in-a-bag/"
leetcode_ui: true
entry_slug: "2024-12-07-1760-minimum-limit-of-balls-in-a-bag"
---

[1760. Minimum Limit of Balls in a Bag](https://leetcode.com/problems/minimum-limit-of-balls-in-a-bag/description/) medium
[blog post](https://leetcode.com/problems/minimum-limit-of-balls-in-a-bag/solutions/6122294/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07122024-1760-minimum-limit-of-balls?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/PPJbq_8MliE)
[deep-dive](https://notebooklm.google.com/notebook/9c676e52-2415-4f5e-b52e-fa1513c6b85a/audio)
![1.webp](/assets/leetcode_daily_images/3699bd39.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/825

#### Problem TLDR

Max number after at most `maxOperations` of splitting #medium #binary_search

#### Intuition

Let's observe the problem:

```j

    // 9 -> 1 8 -> 1 1 7
    //      2 7    2 2 5
    //      3 6    3 3 3 ?? math puzzle
    //      4 5    4 2 3 ??
    // 9 / 2 / 2

    // 9/3 -> 3 3 3
    // 12/3 -> 3 3 3 3 = 3 (3 (3 3))
    // 6/3 -> 3 3
    // 7/3 -> 3 (3 1)
    // 5/3 -> 3 2

```
First (naive) intuition is to try to greedily take the largest number and split it evenly. However, it will not work for the test case `9, maxOps = 2`, which produces `4 2 3` instead of `3 3 3`, giving not optimal result of `4`.

(this is a place where I gave up and used the hint)

The hint is: binary search.

But how can I myself deduce binary search at this point?
Some thoughts:
* problem size: 10^5 numbers, 10^9 max number -> must be linear or nlog(n) at most (but using the problem size is not always an option)
* the task is to maximize/minimize something when there is a constraint like `at most`/`at least` -> maybe it is a function of the constraint and can be searched by it

#### Approach

* pay attention to the values `low` and `high` of the binary search

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minimumSize(nums: IntArray, maxOperations: Int): Int {
        var l = 1; var h = nums.max()
        while (l <= h) {
            val m = (l + h) / 2
            val ops = nums.sumOf { (it - 1) / m }
            if (ops > maxOperations) l = m + 1 else h = m - 1
        }
        return l
    }

```
```rust

    pub fn minimum_size(nums: Vec<i32>, max_operations: i32) -> i32 {
        let (mut l, mut h) = (1, 1e9 as i32);
        while l <= h {
            let m = (l + h) / 2;
            let o: i32 = nums.iter().map(|&x| (x - 1) / m).sum();
            if o > max_operations { l = m + 1 } else { h = m - 1 }
        }; l
    }

```
```c++

    int minimumSize(vector<int>& nums, int maxOperations) {
        int l = 1, h = 1e9;
        while (l <= h) {
            int m = (l + h) / 2, o = 0;
            for (int x: nums) o += (x - 1) / m;
            o > maxOperations ? l = m + 1 : h = m - 1;
        } return l;
    }

```

