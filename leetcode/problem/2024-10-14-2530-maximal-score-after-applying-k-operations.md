---
layout: leetcode-entry
title: "2530. Maximal Score After Applying K Operations"
permalink: "/leetcode/problem/2024-10-14-2530-maximal-score-after-applying-k-operations/"
leetcode_ui: true
entry_slug: "2024-10-14-2530-maximal-score-after-applying-k-operations"
---

[2530. Maximal Score After Applying K Operations](https://leetcode.com/problems/maximal-score-after-applying-k-operations/description/) medium
[blog post](https://leetcode.com/problems/maximal-score-after-applying-k-operations/solutions/5910552/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14102024-2530-maximal-score-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tOYUyFQBJz8)
![1.webp](/assets/leetcode_daily_images/b4aa73f9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/768

#### Problem TLDR

Replace `max(arr)` with `ceil(max/3)` `k` times #medium #heap

#### Intuition

Simulate the process, pick the maximum, add back modified value.
To maintain the sorted order use a `heap`.

#### Approach

* Rust heap is a max-heap, Kotlin is a min-heap
* one small optimization is to keep only `k` values in a heap

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxKelements(nums: IntArray, k: Int): Long =
        PriorityQueue<Int>(nums.map { -it }).run {
            (1..k).fold(0L) { r, _ ->
                r - poll().also { add((it - 2) / 3) }.toLong()
            }
        }

```
```rust

    pub fn max_kelements(nums: Vec<i32>, k: i32) -> i64 {
        let mut bh = BinaryHeap::from(nums);
        (0..k).fold(0, |r, _| {
            let v = bh.pop().unwrap(); bh.push((v + 2) / 3);
            r + v as i64
        })
    }

```
```c++

    long long maxKelements(vector<int>& nums, int k) {
        priority_queue<int> pq(nums.begin(), nums.end());
        long long res = 0;
        while (k--)
            res += pq.top(), pq.push((pq.top() + 2) / 3), pq.pop();
        return res;
    }

```
