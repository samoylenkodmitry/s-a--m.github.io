---
layout: leetcode-entry
title: "719. Find K-th Smallest Pair Distance"
permalink: "/leetcode/problem/2024-08-14-719-find-k-th-smallest-pair-distance/"
leetcode_ui: true
entry_slug: "2024-08-14-719-find-k-th-smallest-pair-distance"
---

[719. Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/description/) hard
[blog post](https://leetcode.com/problems/find-k-th-smallest-pair-distance/solutions/5634024/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14082024-719-find-k-th-smallest-pair?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UKTevw63isw)
![1.webp](/assets/leetcode_daily_images/ef68df90.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/702

#### Problem TLDR

`k`th smallest pairs diff in an array #hard #binary_search #two_pointers

#### Intuition

Let's observe all the possible differences:

```j

    // 1 4 5 6 7 8 9 9 10 10
    //   3 1 1 1 1 1 0  1  0
    //     4 2 2 2 2 1  1  1
    //       5 3 3 3 2  2  1
    //         6 4 4 3  3  2
    //           7 5 4  4  3
    //             8 5  5  4
    //               8  6  5
    //                  9  6
    //                     9

```

The main problem is what to do if `k > nums.size`, as for example `diff=1` has `12` elements: `0 0 1 1 1 1 1 1 1 1 1 1`.

Now, use the `hint`:
* For each `diff` there are growing number of elements, so we can do a Binary Search in a space of `diff = 0..max()`.

To quickly find how many pairs are less than the given diff, we can use a two-pointer technique: move the left pointer until `num[r] - num[l] > diff`, and `r - l` would be the number of pairs.

```j

    // 0 1 2 3 4 5 6 7  8  9
    // 1 4 5 6 7 8 9 9 10 10
    //     l     r            max_diff = mid = 3

```

#### Approach

* for more robust Binary Search: always check the last condition and always move the left or the right pointer

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun smallestDistancePair(nums: IntArray, k: Int): Int {
        nums.sort(); var lo = 0; var hi = 1_000_000
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2; var j = 0
            if (k > nums.indices.sumOf { i ->
                while (nums[j] + mid < nums[i]) j++
                i - j
            }) lo = mid + 1 else hi = mid - 1
        }
        return lo
    }

```
```rust

    pub fn smallest_distance_pair(mut nums: Vec<i32>, k: i32) -> i32 {
        nums.sort_unstable(); let (mut lo, mut hi) = (0, 1_000_000);
        while lo <= hi {
            let (mid, mut count, mut j) = (lo + (hi - lo) / 2, 0, 0);
            for i in 0..nums.len() {
                while nums[j] + mid < nums[i] { j += 1 }
                count += i - j;
            }
            if k > count as i32 { lo = mid + 1 } else { hi = mid - 1 }
        }; lo
    }

```

