---
layout: leetcode-entry
title: "2563. Count the Number of Fair Pairs"
permalink: "/leetcode/problem/2025-04-19-2563-count-the-number-of-fair-pairs/"
leetcode_ui: true
entry_slug: "2025-04-19-2563-count-the-number-of-fair-pairs"
---

[2563. Count the Number of Fair Pairs](https://leetcode.com/problems/count-the-number-of-fair-pairs/description/) medium
[blog post](https://leetcode.com/problems/count-the-number-of-fair-pairs/solutions/6666181/kotlin-rust-by-samoylenkodmitry-6hwl/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19042025-2563-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/0Oj_u8-rEJ8)
![1.webp](/assets/leetcode_daily_images/14be8d7e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/963

#### Problem TLDR

Pairs a + b in lower..upper #medium #two_pointers

#### Intuition

This time I was able to solve it with hints.
Previous time was a fail (5 month ago https://leetcode.com/problems/count-the-number-of-fair-pairs/solutions/6040302/kotlin-rust/).

The hints that helped the most:

* both boundaries move only in a single direction (you have to know how to use them though)

Here is my thougths rundown:

```j
    // 0,1,7,4,4,5 3..6
    //     i
    //     how many visited numbers are in range
    //     3 <= x + a[i] <= 6
    //     3 - a[i] <= x <= 6 - a[i]
    // i
    // expect numbers in range
    // 3 <= a[i] + x <= 6
    // 3 - a[i]..6 - a[i] segment tree?
    //                    sort and binary search
    //
    // 0,1, 7,4,4,5         3..6
    // 3 2 -4
    // 4 3 -3
    // 5 4 -2
    // 6 5 -1

    // total n^2 pairs possible, i < j is irrelevant
    // 0 1 4 4 5 7   two-sum
    // l->          (increase left to make sum bigger)
    //         r->  (increase right to make sum bigger)
    // count of pairs > upper
    // count of pairs < lower

    // -2 -1 0 1 2       0..1
    //       * *
    //     *   *
    //     *     *
    //  *        *

    // 0 1 4 4 5 7   3..6
    // *             3..6       just move l and r, they always go to the left
    //   *           2..5  -1
    //     *        -1..2  -3
    //       *
    //         *    -2..1  -1
    //           *  -4..-1 -2

    // 1 2 5 7 9      11..11

```

My observations:
* i and j positions are irrelevant, we can safely sort
* we can do a binary search (I have failed to implement this in Kotlin)
* we can move both borders left as we go right
* for count lower..upper we may apply count(upper) - count(lower) rule

#### Approach

* let's implement everything and see what's the best

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countFairPairs(n: IntArray, lower: Int, upper: Int): Long {
        n.sort(); val n = n.asList()
        return n.withIndex().sumOf { (i, x) -> 1L *
            max(i + 1, -n.binarySearch { if (it <= upper - x) -1 else 1 } - 1) -
            max(i + 1, -n.binarySearch { if (it < lower - x) -1 else 1 } - 1)
        }
    }

```
```kotlin

    fun countFairPairs(n: IntArray, lower: Int, upper: Int): Long {
        n.sort(); var r = 0L; val n = Array(n.size) { n[it] }
        val cmp = Comparator<Int> { a, b -> if (a < b) -1 else 1 }
        for (i in 0..<n.size) r -=
            Arrays.binarySearch(n, i + 1, n.size, upper - n[i] + 1, cmp) -
            Arrays.binarySearch(n, i + 1, n.size, lower - n[i], cmp)
        return r
    }

```
```kotlin

    fun countFairPairs(n: IntArray, lower: Int, upper: Int): Long {
        n.sort(); var l = n.size - 1; var r = l
        return (0..r).sumOf { i ->
            while (l > i && n[l] + n[i] >= lower) l--
            while (r > l && n[r] + n[i] > upper) r--
            1L * max(i, r) - max(i, l)
        }
    }

```
```kotlin

    fun countFairPairs(n: IntArray, lower: Int, upper: Int): Long {
        n.sort(); var res = 0L; var l = n.size - 1; var r = l
        for (i in 0..r) {
            while (l > i && n[l] + n[i] >= lower) l--
            while (r > l && n[r] + n[i] > upper) r--
            if (r <= i) break; res += r - max(i, l)
        }
        return res
    }

```
```kotlin

    fun countFairPairs(n: IntArray, lower: Int, upper: Int): Long {
        fun cnt(max: Int): Long {
            var res = 0L; var l = 0; var r = n.size - 1
            while (l < r) if (n[l] + n[r] > max) r-- else res += r - l++
            return res
        }
        n.sort(); return cnt(upper) - cnt(lower - 1)
    }

```
```rust

    pub fn count_fair_pairs(mut nums: Vec<i32>, lower: i32, upper: i32) -> i64 {
        nums.sort();
        (0..nums.len()).map(|i|
            nums[..i].partition_point(|&n| n <= upper - nums[i]) -
            nums[..i].partition_point(|&n| n < lower - nums[i])
        ).sum::<usize>() as _
    }

```
```rust

    pub fn count_fair_pairs(mut nums: Vec<i32>, lower: i32, upper: i32) -> i64 {
        fn cnt(nums: &Vec<i32>, max: i32) -> i64 {
            let (mut res, mut l, mut r) = (0, 0, nums.len() - 1);
            while l < r {
                if nums[l] + nums[r] > max { r -= 1 }
                else { res += (r - l) as i64; l += 1 }
            } res
        }
        nums.sort(); cnt(&nums, upper) - cnt(&nums, lower - 1)
    }

```
```rust

    pub fn count_fair_pairs(mut n: Vec<i32>, lower: i32, upper: i32) -> i64 {
        n.sort(); let (mut l, mut r, mut res) = (n.len() - 1, n.len() - 1, 0);
        for i in 0..=r {
            while l > i && n[i] + n[l] >= lower { l -= 1 }
            while r > l && n[i] + n[r] > upper { r -= 1 }
            if r <= i { break }; res += (r - i.max(l)) as i64
        } res
    }

```
```rust

    pub fn count_fair_pairs(mut n: Vec<i32>, lower: i32, upper: i32) -> i64 {
        n.sort(); let (mut l, mut r) = (n.len() - 1, n.len() - 1);
        (0..=r).map(|i| {
            while l > i && n[i] + n[l] >= lower { l -= 1 }
            while r > l && n[i] + n[r] > upper { r -= 1 }
            (i.max(r) - i.max(l)) as i64 }).sum()
    }

```
```c++

    long long countFairPairs(vector<int>& a, int l, int u) {
        sort(begin(a), end(a)); long long r = 0;
        for(int m: array{u, l - 1}) for (int i = 0, j = a.size() - 1; i < j;)
            if (a[i] + a[j] > m) --j;
            else r += (m == u ? 1 : -1) * (j - i++);
        return r;
    }

```

