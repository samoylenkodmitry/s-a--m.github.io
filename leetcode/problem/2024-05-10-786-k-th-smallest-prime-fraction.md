---
layout: leetcode-entry
title: "786. K-th Smallest Prime Fraction"
permalink: "/leetcode/problem/2024-05-10-786-k-th-smallest-prime-fraction/"
leetcode_ui: true
entry_slug: "2024-05-10-786-k-th-smallest-prime-fraction"
---

[786. K-th Smallest Prime Fraction](https://leetcode.com/problems/k-th-smallest-prime-fraction/description/) medium
[blog post](https://leetcode.com/problems/k-th-smallest-prime-fraction/solutions/5138575/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10052024-786-k-th-smallest-prime?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KXXRbKjVgec)
![2024-05-10_10-07.webp](/assets/leetcode_daily_images/44bc39be.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/598

#### Problem TLDR

`k`th `arr[i]/arr[j]`, i < j, arr[i] < arr[j] #medium #heap #binary_search

#### Intuition

The n^2-ish solution is trivial: use PriorityQueue to keep lowest `k` fractions and scan n^2 indices pairs.

The folow up is hard. Let's observe the fractions in the matrix `a/b`:
```j
    //          1   2   3   5   a
    //
    //      5   1/5 2/5 3/5
    //      3   1/3 2/3
    //      2   1/2
    //      b
    //
```
The idea is to for any particular `fraction m` count how many fractions are less than it in O(n) time.
We should invent the way of walking the indices based on observation that fractions grow in both directions of the matrix.
Let's iterate over each `a` value `a = arr[i]`. And for each `a` let's move `b = arr[j]` forward while the current fraction is bigger: we can move it only forward and don't need to backtrack, as if `arr[x]/arr[j] > m` than `arr[x..]/arr[j]` is also `> m`.

```j
    // count less than m = 0.5
    // i=0 1/2 1/3 1/5
    //     j=1 j=2      stop on j=2, count(i=0) = 4-2 = size - j
    // i=1     2/3 2/5
    //         j=2 j=3  stop on j=3, count(i=1) = 4-3 = 1
    // i=2         3/5
    //             j=3 j=4 stop on j=4, count = 0

```
Now, we have a continuous function of `count` that grows with `fraction m in 0..1` and can do a BinarySearch for `k` on it.

#### Approach

This BinarySearch is in `double` space, so we can't just use `m + 1` or `m - 1`, and `lo` must not be equal `hi`.

#### Complexity

- Time complexity:
$$O(n^2log^2(k))$$ for the heap, $$O(nlogn)$$ for the binary search (the search space of `0..1` is quantized by the number of pairs, so n^2, log(n^2) = 2log(n))

- Space complexity:
$$O(k)$$ for the heap, $$O(1)$$ for the binary search

#### Code

```kotlin

    fun kthSmallestPrimeFraction(arr: IntArray, k: Int): IntArray {
        val pq = PriorityQueue<IntArray>(Comparator<IntArray> { a, b ->
            -(a[0] * b[1]).compareTo(b[0] * a[1])
        })
        for (j in arr.indices) for (i in 0..<j) {
            pq += intArrayOf(arr[i], arr[j])
            if (pq.size > k) pq.poll()
        }
        return pq.poll()
    }

```
```rust

    pub fn kth_smallest_prime_fraction(arr: Vec<i32>, k: i32) -> Vec<i32> {
        let (mut lo, mut hi, mut r) = (0.0, 1.0, vec![0, 0]);
        while lo < hi {
            let (m, mut j, mut cnt, mut max) = (lo + (hi - lo) / 2.0, 1, 0, 0.0);
            for i in 0..arr.len() - 1 {
                while j < arr.len() && arr[i] as f64 >= m * arr[j] as f64 { j += 1 }
                let f = if j < arr.len() { arr[i] as f64 / arr[j] as f64 } else { break };
                if f > max { max = f; r = vec![arr[i], arr[j]] }
                cnt += (arr.len() - j) as i32
            }
            if cnt == k { break } else if cnt < k { lo = m } else { hi = m }
        }; r
    }

```

