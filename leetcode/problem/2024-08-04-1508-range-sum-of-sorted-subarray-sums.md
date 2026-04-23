---
layout: leetcode-entry
title: "1508. Range Sum of Sorted Subarray Sums"
permalink: "/leetcode/problem/2024-08-04-1508-range-sum-of-sorted-subarray-sums/"
leetcode_ui: true
entry_slug: "2024-08-04-1508-range-sum-of-sorted-subarray-sums"
---

[1508. Range Sum of Sorted Subarray Sums](https://leetcode.com/problems/range-sum-of-sorted-subarray-sums/description/) meidum
[blog post](https://leetcode.com/problems/range-sum-of-sorted-subarray-sums/solutions/5584473/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04082024-1508-range-sum-of-sorted?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-nliAu95NmI)
![1.webp](/assets/leetcode_daily_images/50d5d4bb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/692

#### Problem TLDR

Sum of `[left..right]` of sorted subarray's sums #medium #heap

#### Intuition

Let's look at the subarrays:

```j
    // 3 2 4 100
    // 3
    //   2
    //     4
    //       100
    // 3 2
    //   2 4
    //     4 100
    // 3 2 4
    //   2 4 100
    // 3 2 4 100
```

Each of them formed as an iteration from some index. Let's put all the iterators into a PriorityQueue and always take the smallest. This can take up to n^2 steps, as `right` can be n^2.

Another solution is from lee215' & voturbac': given the `y = f(x)` value we can in a linear time count how many items are lower than `y`. As `f()` grows with `x` we can use the binary search to find an `x`. The result then will be `f(right) - f(left)`.
To find the lower items count in a linear time, we should prepare the prefixes of the subarray's sums: `b[i] = num[0..i].sum()` and as we summing up those subarrays, go deeper: `c[i] = b[0..i].sum()`.
Then, there is a pattern to find the subarray sum with two pointers: move the lower bound until out of condition, then the sum will be `(i - j) * your_value`. The solution will be O(nlog(n)) and it takes `1ms` in Rust compared to `18ms` heap solution.

#### Approach

As n^2 accepted, let's implement the heap solution. In Rust the BinaryHeap is a max-heap, in Kotin - min-heap

#### Complexity

- Time complexity:
$$O(n^2log(n))$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun rangeSum(nums: IntArray, n: Int, left: Int, right: Int): Int {
        val pq = PriorityQueue<Pair<Int, Int>>(compareBy { it.first })
        for (i in nums.indices) pq += nums[i] to i
        return (1..right).fold(0) { res, j ->
            val (sum, i) = pq.poll()
            if (i < nums.lastIndex) pq += (sum + nums[i + 1]) to (i + 1)
            if (j < left) 0 else (res + sum) % 1_000_000_007
        }
    }

```
```rust

// 18 ms

    pub fn range_sum(nums: Vec<i32>, n: i32, left: i32, right: i32) -> i32 {
        let mut bh = BinaryHeap::new();
        for i in 0..nums.len() { bh.push((-nums[i], i)) }
        (1..=right).fold(0, |res, j| {
            let (sum, i) = bh.pop().unwrap();
            if i < nums.len() - 1 { bh.push((sum - nums[i + 1], i + 1)) }
            if j < left { 0 } else { (res - sum) % 1_000_000_007 }
        })
    }

```
```rust

// lee215' + votrubac' = 1ms

    pub fn range_sum(nums: Vec<i32>, n: i32, left: i32, right: i32) -> i32 {
        let n = n as usize; let mut b = vec![0; n + 1]; let mut c = b.clone();
        for i in 0..n { b[i + 1] = b[i] + nums[i]; c[i + 1] = c[i] + b[i + 1] }
        fn sum_k_sums(b: &[i32], c: &[i32], k: i32, n: usize) -> i32 {
            let (mut l, mut r) = (0, b[n]);
            let mut max_score = 0;
            while l <= r {
                let m = l + (r - l) / 2;
                let (mut i, mut cnt) = (0, 0);
                for j in 0..n+1 {
                    while b[j] - b[i] > m { i += 1 }
                    cnt += (j - i) as i32
                }
                if cnt < k { l = m + 1; max_score = max_score.max(m) }  else { r = m - 1 }
            }
            let (mut res, mut i, mut cnt, mut score) = (0, 0, 0, max_score + 1);
            for j in 0..n+1 {
                while b[j] - b[i] > score { i += 1 }
                res += b[j] * (j as i32 - i as i32 + 1) - (c[j] - (if i > 0 { c[i - 1] } else { 0 }));
                res = res % 1_000_000_007;
                cnt += (j - i) as i32
            }
            res - (cnt - k) * score
        }
        sum_k_sums(&b, &c, right, n) - sum_k_sums(&b, &c, left - 1, n)
    }

```

