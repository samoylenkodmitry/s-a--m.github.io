---
layout: leetcode-entry
title: "2641. Cousins in Binary Tree II"
permalink: "/leetcode/problem/2024-10-23-2641-cousins-in-binary-tree-ii/"
leetcode_ui: true
entry_slug: "2024-10-23-2641-cousins-in-binary-tree-ii"
---

[2641. Cousins in Binary Tree II](https://leetcode.com/problems/cousins-in-binary-tree-ii/description/) medium
[blog post](https://leetcode.com/problems/cousins-in-binary-tree-ii/solutions/5956421/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23102024-2641-cousins-in-binary-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ufs76WuWkfc)
[deep-dive](https://notebooklm.google.com/notebook/88b46d2e-be13-47db-a2e6-f3a8a6dc7def/audio)
![1.webp](/assets/leetcode_daily_images/a99797be.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/777

#### Problem TLDR

Replace Tree's values with cousines sum #medium #bfs

#### Intuition

First, understand the problem, we only care about the current level's row:
![img.jpg](/assets/leetcode_daily_images/568a2c63.webp)

Now, the task is to traverse Tree level by level and precompute the total `next level` sum and the `current parent's` sum.

#### Approach

* consider only the current and the next level
* we can modify at the same time as adding to the queue

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun replaceValueInTree(root: TreeNode?): TreeNode? {
        val q = ArrayDeque<TreeNode>(listOf(root ?: return root))
        while (q.size > 0) {
            val sum = q.sumBy { (it.left?.`val` ?: 0) + (it.right?.`val` ?: 0) }
            repeat(q.size) { q.removeFirst().run {
                var nv = sum - (left?.`val` ?: 0) - (right?.`val` ?: 0)
                left?.let { it.`val` = nv; q += it }
                right?.let { it.`val` = nv; q += it }
            }}
        }
        return root.also { it.`val` = 0 }
    }

```
```rust

    pub fn replace_value_in_tree(mut root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        let Some(r) = root.clone() else { return root }; let mut q = VecDeque::from([r]);
        while q.len() > 0 {
            let mut sum = q.iter().map(|n| { let n = n.borrow();
                n.left.as_ref().map_or(0, |n| n.borrow().val) +
                n.right.as_ref().map_or(0, |n| n.borrow().val)}).sum::<i32>();
            for _ in 0..q.len() {
                let n = q.pop_front().unwrap(); let mut n = n.borrow_mut();
                let mut s =  sum - n.left.as_ref().map_or(0, |n| n.borrow().val) -
                             n.right.as_ref().map_or(0, |n| n.borrow().val);
                if let Some(l) = n.left.clone() { l.borrow_mut().val = s; q.push_back(l); }
                if let Some(r) = n.right.clone() { r.borrow_mut().val = s; q.push_back(r); }
            }
        }
        if let Some(r) = &root { r.borrow_mut().val = 0 }; root
    }

```
```c++

    TreeNode* replaceValueInTree(TreeNode* root) {
        if (!root) return root; queue<TreeNode*> q({root}); root->val = 0;
        while (!q.empty()) {
            int sum = 0, size = q.size();
            for (int i = 0; i < size; ++i, q.push(q.front()), q.pop()) {
                auto node = q.front();
                sum += (node->left ? node->left->val : 0) + (node->right ? node->right->val : 0);
            }
            for (int i = 0; i < size; ++i) {
                auto node = q.front(); q.pop();
                int nv = sum - (node->left ? node->left->val : 0) - (node->right ? node->right->val : 0);
                if (node->left) node->left->val = nv, q.push(node->left);
                if (node->right) node->right->val = nv, q.push(node->right);
            }
        }
        return root;
    }

```

