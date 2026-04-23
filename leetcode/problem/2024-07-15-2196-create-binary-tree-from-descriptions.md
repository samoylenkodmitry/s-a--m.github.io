---
layout: leetcode-entry
title: "2196. Create Binary Tree From Descriptions"
permalink: "/leetcode/problem/2024-07-15-2196-create-binary-tree-from-descriptions/"
leetcode_ui: true
entry_slug: "2024-07-15-2196-create-binary-tree-from-descriptions"
---

[2196. Create Binary Tree From Descriptions](https://leetcode.com/problems/create-binary-tree-from-descriptions/description/) medium
[blog post](https://leetcode.com/problems/create-binary-tree-from-descriptions/description/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15072024-2196-create-binary-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EsKyk_LN9Wk)
![2024-07-15_08-14.webp](/assets/leetcode_daily_images/6f42026c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/671

#### Problem TLDR

Restore binary tree from `[parent, child, isLeft]` #medium

#### Intuition

Use the HashMap. Remember which nodes are children.

#### Approach

* Kotlin: `getOrPut`
* Rust: `entry.or_insert_with`. `Rc` cloning is cheap.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun createBinaryTree(descriptions: Array<IntArray>): TreeNode? {
        val valToNode = mutableMapOf<Int, TreeNode>()
        val children = mutableSetOf<Int>()
        for ((parent, child, isLeft) in descriptions) {
            val pNode = valToNode.getOrPut(parent) { TreeNode(parent) }
            val cNode = valToNode.getOrPut(child) { TreeNode(child) }
            if (isLeft > 0) pNode.left = cNode else pNode.right = cNode
            children += child
        }
        return valToNode.entries.find { it.key !in children }?.value
    }

```
```rust

    pub fn create_binary_tree(descriptions: Vec<Vec<i32>>) -> Option<Rc<RefCell<TreeNode>>> {
        let mut map = HashMap::new(); let mut set = HashSet::new();
        let mut get = |v| { map.entry(v).or_insert_with(|| Rc::new(RefCell::new(TreeNode::new(v)))).clone() };
        for d in descriptions {
            let child = get(d[1]);
            let mut parent = get(d[0]); let mut parent = parent.borrow_mut();
            set.insert(d[1]);
            *(if d[2] > 0 { &mut parent.left } else { &mut parent.right }) = Some(child)
        }
        map.into_values().find(|v| !set.contains(&v.borrow().val))
    }

```

