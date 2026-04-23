---
layout: leetcode-entry
title: "3066. Minimum Operations to Exceed Threshold Value II"
permalink: "/leetcode/problem/2025-02-13-3066-minimum-operations-to-exceed-threshold-value-ii/"
leetcode_ui: true
entry_slug: "2025-02-13-3066-minimum-operations-to-exceed-threshold-value-ii"
---

[3066. Minimum Operations to Exceed Threshold Value II](https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-ii/description/) medium
[blog post](https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-ii/solutions/6416545/kotlin-rust-by-samoylenkodmitry-obbv/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13022025-3066-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/4PF_WjsqiCg)
![1.webp](/assets/leetcode_daily_images/39c1e3bc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/894

#### Problem TLDR

Count nums += min(x,y)*2+max(x,y) < k #medium #heap

#### Intuition

There is only a heap solution.

#### Approach

* some small tricks are possible, given resul is guaranteed by rules
* in-place heap is possible

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minOperations(nums: IntArray, k: Int): Int {
        val q = PriorityQueue(nums.map { 1L * it })
        while (q.peek() < k) q += q.poll() * 2 + q.poll()
        return nums.size - q.size
    }

```
```rust

    pub fn min_operations(nums: Vec<i32>, k: i32) -> i32 {
        let mut q = BinaryHeap::from_iter(nums.iter().map(|x| -x as i64));
        while let Some(x) = q.pop().filter(|&x| x > -k as i64) {
            let y = q.pop().unwrap(); q.push(x * 2 + y)
        }; (nums.len() - q.len() - 1) as i32
    }

```
```c++

    int minOperations(vector<int>& n, int k) {
        priority_queue<long, vector<long>, greater<>> q(begin(n), end(n));
        while (q.top() < k) {
            auto x = 2 * q.top(); q.pop(); x += q.top(); q.pop();
            q.push(x);
        }
        return size(n) - size(q);
    }

```

