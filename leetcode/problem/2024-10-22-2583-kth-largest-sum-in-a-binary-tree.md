---
layout: leetcode-entry
title: "2583. Kth Largest Sum in a Binary Tree"
permalink: "/leetcode/problem/2024-10-22-2583-kth-largest-sum-in-a-binary-tree/"
leetcode_ui: true
entry_slug: "2024-10-22-2583-kth-largest-sum-in-a-binary-tree"
---

[2583. Kth Largest Sum in a Binary Tree](https://leetcode.com/problems/kth-largest-sum-in-a-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/kth-largest-sum-in-a-binary-tree/solutions/5951255/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22102024-2583-kth-largest-sum-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BT8DmUUSt68)
[deep-dive](https://notebooklm.google.com/notebook/f1998c38-e84d-4ab3-b881-27b4982047a6/audio)
![1.webp](/assets/leetcode_daily_images/cd62d518.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/776

#### Problem TLDR

`k`th largest level-sum in a tree #bfs #heap #quickselect

#### Intuition

To collect level sums we can use an iterative Breadth-First Search or a recursive Depth-First Search with level tracking.

To find `k`th largest, we can use a `min-heap` and maintain at most `k` items in it, or we can collect all the sums and then do a `Quickselect` algorithm to find `k`th largest value in O(n)

#### Approach

* it is simpler to store a non-null values in the queue
* in Rust we can destroy the tree with `take` or do a cheap `Rc::clone` (a simple `.clone()` call will do the recursive cloning and is slow)
* in c++ has built-in `nth_element` for Quickselect

#### Complexity

- Time complexity:
$$O(n + log(n)log(k))$$ or O(n) for Quickselect

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun kthLargestLevelSum(root: TreeNode?, k: Int): Long {
        val pq = PriorityQueue<Long>()
        val q = ArrayDeque<TreeNode>(listOf(root ?: return -1))
        while (q.size > 0) {
            pq += (1..q.size).sumOf { q.removeFirst().run {
                left?.let { q += it }; right?.let { q += it }; `val`.toLong() }}
            if (pq.size > k) pq.poll()
        }
        return if (pq.size == k) pq.poll() else -1
    }

```
```rust

    pub fn kth_largest_level_sum(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i64 {
        let Some(r) = root else { return -1i64 };
        let (mut q, mut bh) = (VecDeque::from([r]), BinaryHeap::new());
        while q.len() > 0 {
            let sum = (0..q.len()).map(|_|{
                let n = q.pop_front().unwrap(); let n = n.borrow();
                if let Some(l) = &n.left { q.push_back(Rc::clone(l)) };
                if let Some(r) = &n.right { q.push_back(Rc::clone(r)) };
                n.val as i64
            }).sum::<i64>();
            bh.push(-sum); if bh.len() > k as usize { bh.pop(); }
        }
        if bh.len() == k as usize { -bh.pop().unwrap() } else { -1 }
    }

```
```c++

    long long kthLargestLevelSum(TreeNode* root, int k) {
        queue<TreeNode*>q; q.push(root); vector<long long> s;
        while (!q.empty()) {
            long long sum = 0;
            for (int i = q.size(); i; --i) {
                TreeNode* node = q.front(); q.pop(); sum += node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            s.push_back(sum);
        }
        return s.size() < k ? -1 : (nth_element(begin(s), begin(s) + k - 1, end(s), greater<>()), s[k-1]);
    }

```

