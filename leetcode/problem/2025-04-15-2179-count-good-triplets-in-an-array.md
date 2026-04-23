---
layout: leetcode-entry
title: "2179. Count Good Triplets in an Array"
permalink: "/leetcode/problem/2025-04-15-2179-count-good-triplets-in-an-array/"
leetcode_ui: true
entry_slug: "2025-04-15-2179-count-good-triplets-in-an-array"
---

[2179. Count Good Triplets in an Array](https://leetcode.com/problems/count-good-triplets-in-an-array/description/) hard
[blog post](https://leetcode.com/problems/count-good-triplets-in-an-array/solutions/6652698/kotlin-rust-by-samoylenkodmitry-c8wk/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15042025-2179-count-good-triplets?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cLhuOMkIAoA)
![1.webp](/assets/leetcode_daily_images/2f50b727.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/959

#### Problem TLDR

Triplets in same order in both (0..n)-arrays #hard #bit

#### Intuition

Didn't solve, as I am not familiar with Fenwick Tree or Binary Indexed Tree.

Some of my thought process and observations:

```j

    // 4,0,1,3,2
    // 4,1,0,2,3
    //*4          (0 1 3 2) and (1 0 2 3)
    //   0        (1 3 2) and (2 3) -> 2,3
    //     1      (3 2) and (0 2 3) -> 2,3
    //
    //  *0         (1 3 2) and (2 3)
    //     1-      can't take, pos2(-1) < pos2(0)
    //       3     (2) and ()
    // this is an n^2 algo

    // numbers are exactly 0..n-1

    // 0 1 2 3 4    can we sort both and preserve the relations?
    // 0   1   2
    // 0 3   4

    // 0 1 2 3 4
    // 4,0,1,3,2  -> 0 1 2 3 4
    // 4,1,0,2,3  -> 0 2 1 4 3    now the problem is to count increasing sequencies
    // 0 2 1 4 3       .   .      number of increasing triplets? or subsequences?
    //               0 2   4
    //               0 2   . 3
    //               0 . 1 4
    //               0 . 1 . 3
    //                 .   .      monotonic stack    count smaller than current
    //               0 .   .      0                  0
    //                 2   .      02                 1
    //                   1 .      01                 1
    //                     4      014                2 (lost '2')
    // use the hint - totally different algo
    // the useful hint - triplets are better observed by middle: count smaller * count bigger
    //               0            count less = 0, count bigger = n - 1
    //                 2          count less = 1, count bigger = n - 1 - 1

```

The most helpful observation was that problem can be narrowed down to a single array with increased triplets.

The most helpful hint is for `triplets`: `consider the middle`, then the problem became how much to the `left` and how much to the `right`.

However, to answer `how many numbers are less than current` in a less than O(n^2) you have to know BIT.

BIT:

```j

    //
    // 2 0 1 3 -> 0 1 2 3
    // 0 1 2 3 -> 1 2 0 3
    //
    // didn't quite get how to use BIT here
    // count values smaller/bigger than x
    // add x, remove x

    //                                     16
    //               8                     16
    //       4       8         12          16
    //   2   4   6   8   10    12    14    16
    // 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    //
    // add 6: 7 -> 8 -> 16
    // add 4: 5 -> 6 -> 8 -> 16
    // count less than 8:  9_1001(0) -> 8_1000(2) -> 0
    // count less than 7:  8_1000(2) -> 0
    // count less than 6:  7_111(0) -> 6_110(1) -> 4_100 -> 0
    // count less than 5:  6_110(1) -> 4_100 -> 0
    // count less than 4:  5_101(0) -> 4_100 -> 0

```

This is not the first time I see the BIT, but it is so rare, I forgot how it works.
The idea is the `bits`: `each rightmost bit is a parent of all the left bits`.
The core implementation tricks:
* use idx + 1
* use i & (-i), and `+` makes it go to the parent, `-` iterates all the children

#### Approach

* learn BIT
* steal the solution (u/votrubac/ good solutions)
* count bigger is `n - count smaller`, where n should be adjusted as we go

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun goodTriplets(nums1: IntArray, nums2: IntArray): Long {
        val ids = IntArray(nums1.size); for (i in ids.indices) ids[nums2[i]] = i
        val bt = IntArray(nums1.size + 1); var j = 0
        return (0..<ids.size).sumOf { i ->
            val mid = ids[nums1[i]]; var l = 0L
            j = mid + 1; while (j > 0) { l += bt[j]; j -= j and (-j) }
            j = mid + 1; while (j <= nums1.size) { bt[j]++; j += j and (-j) }
            l * (nums2.size - 1 - mid - (i - l))
        }
    }

```
```rust

    pub fn good_triplets(n1: Vec<i32>, n2: Vec<i32>) -> i64 {
        let (mut ids, mut bt, mut j) = (vec![0; n1.len()], vec![0; n1.len() + 1], 0);
        for i in 0..n1.len() { ids[n2[i] as usize] = i as i64 }; let n = n1.len() as i64;
        (0..n).map(|i| {
            let (mid, mut l) = (ids[n1[i as usize] as usize], 0);
            j = mid + 1; while j > 0 { l += bt[j as usize]; j -= j & (-j) }
            j = mid + 1; while j <= n { bt[j as usize] += 1; j += j & (-j) }
            l * (n - 1 - mid - (i - l))
        }).sum()
    }

```
```c++

    long long goodTriplets(vector<int>& n1, vector<int>& n2) {
        int n = size(n1); vector<int> bt(n + 1), ids(n); long long r = 0;
        for (int i = 0; i < n; ++i) ids[n2[i]] = i;
        for (int i = 0; i < n; ++i) {
            int mid = ids[n1[i]], l = 0, j = 0;
            j = mid + 1; while (j > 0) l += bt[j], j -= j & (-j);
            j = mid + 1; while (j <= n) bt[j]++, j += j & (-j);
            r += 1LL * l * (n - 1 - mid - (i - l));
        } return r;
    }

```

