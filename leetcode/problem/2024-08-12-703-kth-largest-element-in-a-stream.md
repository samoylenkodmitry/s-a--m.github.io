---
layout: leetcode-entry
title: "703. Kth Largest Element in a Stream"
permalink: "/leetcode/problem/2024-08-12-703-kth-largest-element-in-a-stream/"
leetcode_ui: true
entry_slug: "2024-08-12-703-kth-largest-element-in-a-stream"
---

[703. Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/description/) easy
[blog post](https://leetcode.com/problems/kth-largest-element-in-a-stream/solutions/5624559/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12082024-703-kth-largest-element?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Mw92X1fITu8)
![1.webp](/assets/leetcode_daily_images/df5b5152.webp)

#### Problem TLDR

`k`th largest in a stream of values #easy #heap

#### Intuition

Use the heap.

#### Approach

In Kotlin PriorityQueue is a max-heap, in Rust BinaryHeap is a min-heap.

#### Complexity

- Time complexity:
$$O(log(k))$$ for `add` operation, O(nlog(k)) total

- Space complexity:
$$O(k)$$

#### Code

```kotlin

class KthLargest(val k: Int, nums: IntArray) {
    val pq = PriorityQueue<Int>()
    init { for (n in nums) add(n) }
    fun add(v: Int) = pq
        .run { pq += v; if (size > k) poll(); peek() }
}

```
```rust

struct KthLargest { bh: BinaryHeap<i32>, k: usize }
impl KthLargest {
    fn new(k: i32, nums: Vec<i32>) -> Self {
        let mut kth =  Self { bh: BinaryHeap::new(), k: k as usize };
        for &n in nums.iter() { kth.add(n); }
        kth
    }
    fn add(&mut self, val: i32) -> i32 {
        self.bh.push(-val);
        if self.bh.len() > self.k { self.bh.pop(); }
        -self.bh.peek().unwrap()
    }
}

```

