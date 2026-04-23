---
layout: leetcode-entry
title: "2948. Make Lexicographically Smallest Array by Swapping Elements"
permalink: "/leetcode/problem/2025-01-25-2948-make-lexicographically-smallest-array-by-swapping-elements/"
leetcode_ui: true
entry_slug: "2025-01-25-2948-make-lexicographically-smallest-array-by-swapping-elements"
---

[2948. Make Lexicographically Smallest Array by Swapping Elements](https://leetcode.com/problems/make-lexicographically-smallest-array-by-swapping-elements/description/) medium
[blog post](https://leetcode.com/problems/make-lexicographically-smallest-array-by-swapping-elements/solutions/6327211/kotlin-rust-by-samoylenkodmitry-7avo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25012025-2948-make-lexicographically?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/lFQyQTgf3bg)
![1.webp](/assets/leetcode_daily_images/7662fda2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/875

#### Problem TLDR

Sort by swapping ab, where abs(a - b) < limit #medium

#### Intuition

Let's observe an example:

```j

    // 0 1 2  3 4 5
    // 1 7 6 18 2 1
    // *        * * (1..2)
    //   * *        (6..7)
    //        *     (18..18)

    // 0 5 4 2 1  3
    // 1 1 2 6 7 18
    // * * *
    //       * *
    //            *

```

We have a separate groups that can be sorted. One way to find `gaps > limit` is to sort the array and scan it linearly.

#### Approach

* we can use a Heap, or just sort

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun lexicographicallySmallestArray(nums: IntArray, limit: Int): IntArray {
        val ix = nums.indices.sortedBy { nums[it] }; var j = 0
        val qi = PriorityQueue<Int>(); val res = IntArray(nums.size)
        for (i in ix.indices) {
            qi += ix[i]
            if (i == ix.size - 1 || nums[ix[i + 1]] - nums[ix[i]] > limit)
                while (qi.size > 0) res[qi.poll()] = nums[ix[j++]]
        }
        return res
    }

```
```rust

    pub fn lexicographically_smallest_array(nums: Vec<i32>, limit: i32) -> Vec<i32> {
        let mut ix: Vec<_> = (0..nums.len()).collect(); ix.sort_by_key(|&x| nums[x]);
        let (mut h, mut r, mut j) = (BinaryHeap::new(), vec![0; ix.len()], 0);
        for i in 0..ix.len() {
            h.push(Reverse(ix[i]));
            if i == ix.len() - 1 || nums[ix[i + 1]] - nums[ix[i]] > limit {
                while let Some(Reverse(k)) = h.pop() { r[k] = nums[ix[j]]; j += 1 }
            }
        }; r
    }

```
```c++

    vector<int> lexicographicallySmallestArray(vector<int>& nums, int limit) {
        vector<int> ix(size(nums)), r(size(nums)); iota(begin(ix), end(ix), 0);
        sort(begin(ix), end(ix), [&](int a, int b) { return nums[a] < nums[b]; });
        priority_queue<int, vector<int>, greater<>> q;
        for (int i = 0, j = 0; i < size(ix); ++i) {
            q.push(ix[i]);
            if (i == size(ix) - 1 || nums[ix[i + 1]] - nums[ix[i]] > limit)
                while (size(q)) r[q.top()] = nums[ix[j++]], q.pop();
        } return r;
    }

```

