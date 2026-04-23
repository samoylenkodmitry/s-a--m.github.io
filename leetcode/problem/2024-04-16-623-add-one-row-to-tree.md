---
layout: leetcode-entry
title: "623. Add One Row to Tree"
permalink: "/leetcode/problem/2024-04-16-623-add-one-row-to-tree/"
leetcode_ui: true
entry_slug: "2024-04-16-623-add-one-row-to-tree"
---

[623. Add One Row to Tree](https://leetcode.com/problems/add-one-row-to-tree/description/) medium
[blog post](https://leetcode.com/problems/add-one-row-to-tree/solutions/5030293/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16042024-623-add-one-row-to-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/S9jxRF_mtHY)
![2024-04-16_08-54.webp](/assets/leetcode_daily_images/63b312ed.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/573

#### Problem TLDR

Insert nodes at the `depth` of the Binary Tree #medium

#### Intuition

We can use Depth-First or Breadth-First Search

#### Approach

Let's use DFS in Kotlin, and BFS in Rust.
In a DFS solution we can try to use result of a function to shorten the code: to identify which node is right, mark depth as zero for it.

#### Complexity

- Time complexity:
$$O(n)$$, for both DFS and BFS

- Space complexity:
$$O(log(n))$$ for DFS, but O(n) for BFS as the last row can contain as much as `n/2` items

#### Code

```kotlin

    fun addOneRow(root: TreeNode?, v: Int, depth: Int): TreeNode? =
        if (depth < 2) TreeNode(v).apply { if (depth < 1) right = root else left = root }
        else root?.apply {
            left = addOneRow(left, v, depth - 1)
            right = addOneRow(right, v, if (depth < 3) 0 else depth - 1)
        }

```
```rust

    pub fn add_one_row(mut root: Option<Rc<RefCell<TreeNode>>>, val: i32, depth: i32) -> Option<Rc<RefCell<TreeNode>>> {
        if depth < 2 { return Some(Rc::new(RefCell::new(TreeNode { val: val, left: root, right: None }))) }
        let mut queue = VecDeque::new(); queue.push_back(root.clone());
        for _ in 2..depth { for _ in 0..queue.len() {
                if let Some(n) = queue.pop_front() { if let Some(n) = n {
                        let n = n.borrow();
                        queue.push_back(n.left.clone());
                        queue.push_back(n.right.clone());
                } }
        } }
        while queue.len() > 0 {
            if let Some(n) = queue.pop_front() { if let Some(n) = n {
                    let mut n = n.borrow_mut();
                    n.left = Some(Rc::new(RefCell::new(TreeNode { val: val, left: n.left.take(), right: None })));
                    n.right = Some(Rc::new(RefCell::new(TreeNode { val: val, left: None, right: n.right.take() })));
            } }
        }; root
    }

```

